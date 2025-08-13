//Hidden state is now a vector of 3 values → more “memory neurons” to store past info

//Weights are now matrices instead of scalars

//We do matrix–vector multiplications to update the hidden state


// Helper: create matrix with given shape filled with random small values
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

// Helper: matrix-vector multiplication
function matVecMul(mat, vec) {
    let result = new Array(mat.length).fill(0);
    for (let r = 0; r < mat.length; r++) {
        for (let c = 0; c < vec.length; c++) {
            result[r] += mat[r][c] * vec[c];
        }
    }
    return result;
}

// Helper: vector addition
function vecAdd(a, b) {
    return a.map((v, i) => v + b[i]);
}

// Helper: apply tanh to each element
function tanhVec(vec) {
    return vec.map(v => Math.tanh(v));
}

// Helper: dot product of vectors
function dot(a, b) {
    return a.reduce((sum, v, i) => sum + v * b[i], 0);
}

// RNN parameters
let inputSize = 1;
let hiddenSize = 3;
let outputSize = 1;

let Wxh = randomMatrix(hiddenSize, inputSize);  // input → hidden
let Whh = randomMatrix(hiddenSize, hiddenSize); // hidden → hidden
let Why = new Array(hiddenSize).fill(0).map(() => Math.random() * 0.2 - 0.1); // hidden → output

let bh = new Array(hiddenSize).fill(0); // hidden bias
let by = 0; // output bias

// Forward step for one time step
function rnnStep(input, hiddenPrev) {
    let inputVec = [input]; // wrap into array for matrix math

    // input to hidden
    let h1 = matVecMul(Wxh, inputVec);
    // hidden to hidden
    let h2 = matVecMul(Whh, hiddenPrev);
    // add and activate
    let hidden = tanhVec(vecAdd(vecAdd(h1, h2), bh));

    // hidden to output (dot product)
    let output = dot(hidden, Why) + by;

    return { hidden, output };
}

// Test the RNN on a sequence
let inputs = [1, 2, 3, 4];
let hiddenState = new Array(hiddenSize).fill(0);

console.log("RNN Predictions:");
for (let t = 0; t < inputs.length; t++) {
    let step = rnnStep(inputs[t], hiddenState);
    hiddenState = step.hidden; // update memory
    console.log(`Input: ${inputs[t]} → Output: ${step.output.toFixed(4)}`);
}
