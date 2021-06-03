//const tf = require('@tensorflow/tfjs-node');
const tf = require('@tensorflow/tfjs');
const tfvis = require('@tensorflow/tfjs-vis');

console.log("TensorFlow.js version information: ");
console.log(tf.version);

console.log(`TensorFlow.js backend: ${tf.getBackend()}`);
