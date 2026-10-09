const TRAINING_DATA_URL = "train_data.csv";
const LABEL_NAMES = { "0": "Negative", "2": "Positive" };

function parseCsv(source) {
  source = source.replace(/^\uFEFF/, "");
  const rows = [];
  let row = [];
  let field = "";
  let quoted = false;

  for (let index = 0; index < source.length; index += 1) {
    const character = source[index];

    if (quoted) {
      if (character === '"' && source[index + 1] === '"') {
        field += '"';
        index += 1;
      } else if (character === '"') {
        quoted = false;
      } else {
        field += character;
      }
    } else if (character === '"' && field.length === 0) {
      quoted = true;
    } else if (character === ",") {
      row.push(field);
      field = "";
    } else if (character === "\n" || character === "\r") {
      row.push(field);
      rows.push(row);
      row = [];
      field = "";
      if (character === "\r" && source[index + 1] === "\n") index += 1;
    } else {
      field += character;
    }
  }

  if (field.length > 0 || row.length > 0) {
    row.push(field);
    rows.push(row);
  }
  return rows;
}

function tokenize(text) {
  const normalized = text.normalize("NFD").replace(/\p{Diacritic}/gu, "").toLowerCase();
  return normalized.match(/[a-z0-9]+(?:'[a-z0-9]+)?/g) || [];
}

function featuresFor(text) {
  const words = tokenize(text);
  const features = [];
  for (let index = 0; index < words.length; index += 1) {
    features.push(`u:${words[index]}`);
    if (index > 0) features.push(`b:${words[index - 1]}_${words[index]}`);
  }
  return features;
}

async function trainModel(source) {
  const rows = parseCsv(source);
  if (rows.length < 2 || rows[0].length < 2) {
    throw new Error("The training file is empty or has an invalid CSV format.");
  }

  const textColumn = rows[0].indexOf("text");
  const labelColumn = rows[0].indexOf("label");
  if (textColumn < 0 || labelColumn < 0) {
    throw new Error("The training CSV must contain 'text' and 'label' columns.");
  }

  const counts = { "0": new Map(), "2": new Map() };
  const documentCounts = { "0": 0, "2": 0 };
  const tokenCounts = { "0": 0, "2": 0 };
  const totalFeatureCounts = new Map();
  let examples = 0;

  for (let rowIndex = 1; rowIndex < rows.length; rowIndex += 1) {
    const row = rows[rowIndex];
    const text = (row[textColumn] || "").trim();
    const label = (row[labelColumn] || "").trim();
    if (!text || !Object.hasOwn(LABEL_NAMES, label)) continue;

    const features = featuresFor(text);
    if (features.length === 0) continue;
    documentCounts[label] += 1;
    tokenCounts[label] += features.length;
    for (const feature of features) {
      counts[label].set(feature, (counts[label].get(feature) || 0) + 1);
      totalFeatureCounts.set(feature, (totalFeatureCounts.get(feature) || 0) + 1);
    }
    examples += 1;

    if (examples % 3000 === 0) {
      setStatus(`Training the model... ${examples.toLocaleString()} examples`);
      await new Promise((resolve) => setTimeout(resolve, 0));
    }
  }

  if (documentCounts["0"] === 0 || documentCounts["2"] === 0) {
    throw new Error("The training file must contain examples for both sentiment classes.");
  }

  const vocabularySize = [...totalFeatureCounts.values()].filter((count) => count >= 2).length;
  return {
    counts,
    documentCounts,
    tokenCounts,
    totalDocuments: documentCounts["0"] + documentCounts["2"],
    vocabularySize,
    examples,
  };
}

function predict(model, text) {
  const features = featuresFor(text);
  const vocabularySize = Math.max(model.vocabularySize, 1);
  const scores = {};

  for (const label of ["0", "2"]) {
    scores[label] = Math.log(model.documentCounts[label] / model.totalDocuments);
    const denominator = model.tokenCounts[label] + vocabularySize;
    for (const feature of features) {
      const positiveCount = model.counts["2"].get(feature) || 0;
      const negativeCount = model.counts["0"].get(feature) || 0;
      if (positiveCount + negativeCount < 2) continue;
      scores[label] += Math.log(((model.counts[label].get(feature) || 0) + 1) / denominator);
    }
  }

  const difference = Math.max(-700, Math.min(700, scores["2"] - scores["0"]));
  const positiveProbability = 1 / (1 + Math.exp(-difference));
  return {
    label: positiveProbability >= 0.5 ? "2" : "0",
    confidence: Math.max(positiveProbability, 1 - positiveProbability) * 100,
  };
}

function setStatus(message) {
  const status = document.querySelector("#status");
  if (status) status.textContent = message;
}

async function startApp() {
  const button = document.querySelector("#analyze-button");
  const form = document.querySelector("#sentiment-form");
  const commentInput = document.querySelector("#comment");
  const result = document.querySelector("#result");
  const sentimentLabel = document.querySelector("#sentiment-label");
  const confidence = document.querySelector("#confidence");

  try {
    const response = await fetch(TRAINING_DATA_URL);
    if (!response.ok) throw new Error(`Could not load training data (HTTP ${response.status}).`);
    const model = await trainModel(await response.text());
    button.disabled = false;
    button.textContent = "Analyze sentiment";
    setStatus(`Ready. Trained on ${model.examples.toLocaleString()} positive and negative comments.`);

    form.addEventListener("submit", (event) => {
      event.preventDefault();
      const text = commentInput.value.trim();
      if (!text) return;

      const prediction = predict(model, text);
      sentimentLabel.textContent = LABEL_NAMES[prediction.label];
      sentimentLabel.className = `result-label ${prediction.label === "2" ? "positive" : "negative"}`;
      confidence.textContent = `Model confidence: ${prediction.confidence.toFixed(1)}%`;
      result.hidden = false;
    });
  } catch (error) {
    setStatus(`Unable to start the model: ${error.message}`);
    button.disabled = true;
  }
}

startApp();
