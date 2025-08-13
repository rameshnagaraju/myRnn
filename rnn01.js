// A tiny RNN example in plain JavaScript
// Predicts the next number in a sequence

// Initialize weights randomly
function randomWeight() {
    return Math.random() * 0.2 - 0.1; // small values between -0.1 and 0.1
}

// RNN parameters
let inputSize = 1;
let hiddenSize = 1;
let outputSize = 1;

let Wxh = randomWeight(); // input → hidden
let Whh = randomWeight(); // hidden → hidden (memory loop)
let Why = randomWeight(); // hidden → output

let bh = 0; // hidden bias
let by = 0; // output bias

// Activation function
function tanh(x) {
    return Math.tanh(x);
}

// Forward step for one time step
function rnnStep(input, hiddenPrev) {
    // Calculate new hidden state
    
    let hidden = tanh(Wxh * input + Whh * hiddenPrev + bh);
    // Calculate output
    let output = hidden * Why + by;
    return { hidden, output };
}

// Test the RNN on a sequence
let inputs = [1, 2, 3, 4];
let hiddenState = 0;

console.log("RNN Predictions:");
for (let t = 0; t < inputs.length; t++) {
    let step = rnnStep(inputs[t], hiddenState);
    hiddenState = step.hidden; // update memory
    console.log(`Input: ${inputs[t]} → Output: ${step.output.toFixed(4)}`);
}
