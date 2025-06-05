let lossCount = 0;
const totalLossesExpected = 6;

function showOverlay(id) {
  const el = document.getElementById("overlay-" + id);
  if (el) el.style.display = "flex";
}

function hideOverlay(id) {
  const el = document.getElementById("overlay-" + id);
  if (el) el.style.display = "none";
}

function computeAndDisplayLoss(model, data, label) {
  const xs = tf.tensor2d(data.map(p => [p.x]));
  const ysTrue = tf.tensor2d(data.map(p => [p.y]));
  const ysPred = model.predict(xs);
  const lossTensor = tf.losses.meanSquaredError(ysTrue, ysPred);
  const loss = lossTensor.dataSync()[0];

  const table = document.querySelector("#loss-values tbody");
  const row = document.createElement("tr");

  const [modelName, dataType] = label.split(" ");
  const isNoisy = label.includes("Best-Fit") || label.includes("Overfit");
  const noiseLabel = isNoisy ? "verrauscht" : "unverrauscht";

  row.innerHTML = `
    <td>${modelName}</td>
    <td>${dataType}</td>
    <td>${noiseLabel}</td>
    <td>${loss.toFixed(5)}</td>
  `;

  table.appendChild(row);

  lossCount++;
  if (lossCount === totalLossesExpected) {
    const loadingRow = document.getElementById("loss-loading-row");
    if (loadingRow) loadingRow.remove();
  }

  xs.dispose();
  ysTrue.dispose();
  ysPred.dispose();
  lossTensor.dispose();
}

function plotPrediction(canvasId, model, dataPoints, title) {
  showOverlay(canvasId);

  const xs = dataPoints.map(p => p.x);
  const xsTensor = tf.tensor2d(xs.map(x => [x]));
  const predsTensor = model.predict(xsTensor);
  const preds = Array.from(predsTensor.dataSync());

  const ctx = document.getElementById(canvasId).getContext("2d");

  const chartData = {
    datasets: [
      {
        label: "Echte Werte",
        data: dataPoints.map(p => ({ x: p.x, y: p.y })),
        backgroundColor: "rgba(99, 255, 132, 0.6)",
        pointRadius: 4
      },
      {
        label: "Vorhersage",
        data: xs.map((x, i) => ({ x, y: preds[i] })),
        backgroundColor: "rgba(0, 0, 0, 0.7)",
        pointRadius: 3
      }
    ]
  };

  new Chart(ctx, {
    type: "scatter",
    data: chartData,
    options: {
      responsive: true,
      plugins: {
        title: {
          display: true,
          text: title,
          font: { size: 16 }
        },
        legend: {
          display: true,
          position: "top"
        }
      },
      scales: {
        x: {
          type: "linear",
          title: { display: true, text: "x" }
        },
        y: {
          title: { display: true, text: "y" }
        }
      }
    }
  });

  xsTensor.dispose();
  predsTensor.dispose();

  hideOverlay(canvasId);
}
// Funktion zur Berechnung der Ground-Truth-Funktion y(x)
function trueFunction(x) {
  return 0.5 * (x + 0.8) * (x + 1.8) * (x - 0.2) * (x - 0.3) * (x - 1.9) + 1;
}

// Hilfsfunktion zur Erzeugung von Gaußschem Rauschen
function gaussianNoise(mean = 0, variance = 0.05) {
  const u1 = Math.random();
  const u2 = Math.random();
  const z0 = Math.sqrt(-2.0 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
  return z0 * Math.sqrt(variance) + mean;
}

// Hilfsfunktion zum Mischen eines Arrays (Fisher-Yates)
function shuffle(array) {
  for (let i = array.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [array[i], array[j]] = [array[j], array[i]];
  }
  return array;
}

// Datengenerierung für N = 100
function generateData() {
  const N = 100;
  const xValues = Array.from({ length: N }, () => Math.random() * 4 - 2);
  const dataClean = xValues.map(x => ({ x, y: trueFunction(x) }));
  const dataNoisy = dataClean.map(({ x, y }) => ({ x, y: y + gaussianNoise() }));

  // Mischen und Aufteilen
  const shuffledClean = shuffle([...dataClean]);
  const shuffledNoisy = shuffle([...dataNoisy]);

  const trainClean = shuffledClean.slice(0, 50);
  const testClean = shuffledClean.slice(50);
  const trainNoisy = shuffledNoisy.slice(0, 50);
  const testNoisy = shuffledNoisy.slice(50);

  return {
    clean: { train: trainClean, test: testClean },
    noisy: { train: trainNoisy, test: testNoisy }
  };
}

function plotDataset(canvasId, dataset, title) {
  const ctx = document.getElementById(canvasId).getContext("2d");
  const data = {
    datasets: [
      {
        label: "Trainingsdaten",
        data: dataset.train.map(p => ({ x: p.x, y: p.y })),
        backgroundColor: "rgba(54, 162, 235, 0.7)",
        pointRadius: 4
      },
      {
        label: "Testdaten",
        data: dataset.test.map(p => ({ x: p.x, y: p.y })),
        backgroundColor: "rgba(255, 99, 132, 0.7)",
        pointRadius: 4
      }
    ]
  };

  new Chart(ctx, {
    type: "scatter",
    data: data,
    options: {
      responsive: true,
      plugins: {
        title: {
          display: true,
          text: title,
          font: { size: 16 }
        },
        legend: {
          display: true,
          position: "top"
        }
      },
      scales: {
        x: {
          type: "linear",
          position: "bottom",
          title: {
            display: true,
            text: "x"
          }
        },
        y: {
          title: {
            display: true,
            text: "y"
          }
        }
      }
    }
  });
}

function createModel() {
  const model = tf.sequential();

  // Eingabeschicht + 1. Hidden Layer
  model.add(tf.layers.dense({ units: 100, activation: 'relu', inputShape: [1] }));
  // 2. Hidden Layer
  model.add(tf.layers.dense({ units: 100, activation: 'relu' }));
  // Output Layer
  model.add(tf.layers.dense({ units: 1 })); // linear activation = default

  model.compile({
    optimizer: tf.train.adam(0.01),
    loss: 'meanSquaredError'
  });

  return model;
}

async function trainModel(model, trainData, epochs = 100) {
  const xs = tf.tensor2d(trainData.map(p => [p.x]));
  const ys = tf.tensor2d(trainData.map(p => [p.y]));

  await model.fit(xs, ys, {
    epochs: epochs,
    batchSize: 32,
    shuffle: true,
    callbacks: {
      onEpochEnd: (epoch, logs) => {
        console.log(`Epoch ${epoch + 1} / ${epochs}: Loss = ${logs.loss.toFixed(5)}`);
      }
    }
  });

  xs.dispose();
  ys.dispose();
}

document.addEventListener("DOMContentLoaded", () => {
  const data = generateData();
  plotDataset("dataset-clean", data.clean, "Daten ohne Rauschen");
  plotDataset("dataset-noisy", data.noisy, "Daten mit Rauschen");

  // Modell für unverrauschte Daten
  const modelClean = createModel();
  trainModel(modelClean, data.clean.train, 100).then(() => {
    // Vorhersage für Trainings- und Testdaten anzeigen
    plotPrediction("prediction-clean-train", modelClean, data.clean.train, "Vorhersage (Trainingsdaten, unverrauscht)");
    plotPrediction("prediction-clean-test", modelClean, data.clean.test, "Vorhersage (Testdaten, unverrauscht)");
    computeAndDisplayLoss(modelClean, data.clean.train, "Clean Train");
    computeAndDisplayLoss(modelClean, data.clean.test, "Clean Test");

    // Modell für verrauschte Daten – Best-Fit
    const modelBest = createModel();
    trainModel(modelBest, data.noisy.train, 50).then(() => {
      plotPrediction("prediction-best-train", modelBest, data.noisy.train, "Best-Fit Vorhersage (Trainingsdaten, verrauscht)");
      plotPrediction("prediction-best-test", modelBest, data.noisy.test, "Best-Fit Vorhersage (Testdaten, verrauscht)");
      computeAndDisplayLoss(modelBest, data.noisy.train, "Best-Fit Train");
      computeAndDisplayLoss(modelBest, data.noisy.test, "Best-Fit Test");
    });

    // Modell für verrauschte Daten – Overfit
    const modelOverfit = createModel();
    trainModel(modelOverfit, data.noisy.train, 300).then(() => {
      plotPrediction("prediction-overfit-train", modelOverfit, data.noisy.train, "Overfit Vorhersage (Trainingsdaten, verrauscht)");
      plotPrediction("prediction-overfit-test", modelOverfit, data.noisy.test, "Overfit Vorhersage (Testdaten, verrauscht)");
      computeAndDisplayLoss(modelOverfit, data.noisy.train, "Overfit Train");
      computeAndDisplayLoss(modelOverfit, data.noisy.test, "Overfit Test");
    });
  });
});