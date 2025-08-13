// Simple RNN with training in JavaScript (hidden size = 3)
// Goal: Learn to predict the next number in a sequence
//Forward pass — Predict the next number given the current number + hidden memory

//Backward pass — Compute gradients manually (no TensorFlow magic)

//Update weights with gradient descent

//Repeat for many epochs until loss is small
// ---------------- Helper Functions ----------------

// Random small value matrix
function randomMatrix(rows, cols) {
    let m = [];
    for (let r = 0; r < rows; r++) {
        m[r] = [];
        for (let c = 0; c < cols; c++) {
            m[r][c] = Math.random() * 0.2 - 0.1;
        }
    }
    return m;
}

// Matrix-vector multiplication
function matVecMul(mat, vec) {
    let result = new Array(mat.length).fill(0);
    for (let r = 0; r < mat.length; r++) {
        for (let c = 0; c < vec.length; c++) {
            result[r] += mat[r][c] * vec[c];
        }
    }
    return result;
}

// Vector addition
function vecAdd(a, b) {
    return a.map((v, i) => v + b[i]);
}

// Apply function to each element
function vecMap(vec, fn) {
    return vec.map(fn);
}

// Dot product
function dot(a, b) {
    return a.reduce((sum, v, i) => sum + v * b[i], 0);
}

// Tanh and derivative
function tanhVec(vec) {
    return vec.map(v => Math.tanh(v));
}
function tanhDerivVec(vec) {
    return vec.map(v => 1 - v * v); // derivative given tanh output
}

// ---------------- RNN Parameters ----------------

let inputSize = 1;
let hiddenSize = 3;
let outputSize = 1;
let learningRate = 0.05;

let Wxh = randomMatrix(hiddenSize, inputSize);
let Whh = randomMatrix(hiddenSize, hiddenSize);
let Why = new Array(hiddenSize).fill(0).map(() => Math.random() * 0.2 - 0.1);

let bh = new Array(hiddenSize).fill(0);
let by = 0;

// ---------------- Forward Step ----------------

function rnnStep(input, hiddenPrev) {
    let inputVec = [input];
    let h1 = matVecMul(Wxh, inputVec);
    let h2 = matVecMul(Whh, hiddenPrev);
    let hiddenRaw = vecAdd(vecAdd(h1, h2), bh);
    let hidden = tanhVec(hiddenRaw);
    let output = dot(hidden, Why) + by;

    return { hidden, output, hiddenRaw };
}

// ---------------- Training Loop ----------------

let trainingData = [];
for (let i = 0; i < 20; i++) {
    trainingData.push(i / 20); // numbers from 0.0 to 0.95 step 0.05
}

for (let epoch = 0; epoch < 200; epoch++) {
    let totalLoss = 0;
    let hiddenState = new Array(hiddenSize).fill(0);

    for (let t = 0; t < trainingData.length - 1; t++) {
        let x = trainingData[t];
        let target = trainingData[t + 1];

        // Forward
        let step = rnnStep(x, hiddenState);

        // Loss
        let error = step.output - target;
        totalLoss += error ** 2;

        // Backprop through output
        let dWhy = step.hidden.map(h => h * error);
        let dby = error;

        // Backprop into hidden
        let dhidden = Why.map(w => w * error);
        let dhiddenRaw = dhidden.map((dh, i) => dh * (1 - step.hidden[i] ** 2));

        // Update weights & biases
        for (let i = 0; i < hiddenSize; i++) {
            for (let j = 0; j < inputSize; j++) {
                Wxh[i][j] -= learningRate * dhiddenRaw[i] * x;
            }
            for (let j = 0; j < hiddenSize; j++) {
                Whh[i][j] -= learningRate * dhiddenRaw[i] * hiddenState[j];
            }
            Why[i] -= learningRate * dWhy[i];
            bh[i] -= learningRate * dhiddenRaw[i];
        }
        by -= learningRate * dby;

        // Move to next step
        hiddenState = step.hidden;
    }

    if (epoch % 20 === 0) {
        console.log(`Epoch ${epoch} — Loss: ${(totalLoss / trainingData.length).toFixed(6)}`);
    }
}

// ---------------- Test After Training ----------------
console.log("\nPredictions after training:");
let hiddenState = new Array(hiddenSize).fill(0);
let start = trainingData[0];
let value = start;

for (let i = 0; i < 10; i++) {
    let step = rnnStep(value, hiddenState);
    hiddenState = step.hidden;
    console.log(`Input: ${value.toFixed(2)} → Predicted: ${step.output.toFixed(4)}`);
    value = step.output; // feed prediction back in
}
