console.log('Creating simple linear regression model using TFJS');

const tf = require('@tensorflow/tfjs');

// create a sequential model and add layers to it.
function create_model(){
  const model = tf.sequential();
  model.add(
    tf.layers.dense({
      units: 1,
      useBias: true,
      activation: 'linear',
      inputDim: 1
    }) );

  const optimizer = tf.train.sgd (learningRate=0.1);
  model.compile({
    // Computes the mean squared error between two tensors.
    loss: 'meanSquaredError',
    optimizer
  });
  return model;
}


const M = create_model();
// insepct the model.
M.summary();

const L1 = M.getLayer(undefined, 0);
console.log(L1);
