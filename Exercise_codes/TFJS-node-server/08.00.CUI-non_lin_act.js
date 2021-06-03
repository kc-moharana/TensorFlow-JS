console.log('Design a Commandline User Interface to accept a house_space from user and predict the price.');
console.log('===========');

const inquirer = require('inquirer');
const tf = require('@tensorflow/tfjs');

// global variables (doesnt work in NodeJS, only in Browser mode)
// let model;
// let normalized_feature_tensor, normalized_label_tensor;
// let train_feature_tensor, test_feature_tensor, train_label_tensor, test_label_tensor;

// ======================================================================
function normalize_tensor(tensor, user_min=null, user_max=null){
  const min = user_min || tensor.min();
  const max = user_max || tensor.max();
  return {
    tensor: tensor.sub(min).div(max.sub(min)),
    min,
    max
  }
}

function denormalize_tensor(tensor, min, max) {
  const denormalisedTensor = tensor.mul(max.sub(min)).add(min);
  return denormalisedTensor;
}

async function create_model(){
  const model = tf.sequential();
  model.add(tf.layers.dense({
    activation: 'sigmoid', // Activation function to use.
    useBias: true,        // Whether to apply a bias.
    units: 10,             // Positive integer, dimensionality of the output space.
    inputDim :1           //If specified, defines inputShape as [inputDim].
  }));
  model.add(tf.layers.dense({
    activation: 'sigmoid', // Activation function to use.
    useBias: true,        // Whether to apply a bias.
    units: 10,             // Positive integer, dimensionality of the output space.
  }));
  model.add(tf.layers.dense({
    activation: 'sigmoid', // Activation function to use.
    useBias: true,        // Whether to apply a bias.
    units: 1,             // Positive integer, dimensionality of the output space.
  }));
// const learningRate = 0.1;
// const optimizer = tf.train.sgd(learningRate);
const optimizer = tf.train.adam();


// Compiling outfits the model with an optimizer, loss, and/or metrics.
// Configures and prepares the model for training and evaluation.
model.compile({
  loss: 'meanSquaredError', //The loss value that will be minimized by the model will then be the sum of all individual losses.
  optimizer
});
return model;
}

async function train_model(model,train_feature_tensor, train_label_tensor){
  return model.fit(train_feature_tensor, train_label_tensor,
    {
      batchSize : 32,
      epochs : 100,
      shuffle : true,
      validationSplit: 0.2,
      callbacks:{
        onEpochEnd: (epoch, log)=>console.log(`Epoch ${epoch}: ${log.loss}`)
      }
    });
}

async function test_model(model, test_feature_tensor, test_label_tensor){
  const test_loss_output = model.evaluate(test_feature_tensor, test_label_tensor);
  const test_loss = await test_loss_output.dataSync()[0]
  console.log(`#\tTest set loss ${test_loss.toPrecision(5)}`);
}

async function load_data(){
  const csv_url = 'file://C:/Users/kcm/Documents/GIT_repos/TensorFlow-JS/Exercise_codes/TFJS-node-server/data/kc_house_data.csv';
  const CSVDataset = tf.data.csv(csv_url);
  //console.log(await CSVDataset.columnNames() );
  const data_table = CSVDataset.map(records=>({
    x: records.sqft_living ,
    y: records.price
  }));
  //console.log(await data_table.take(10).toArray());
  const points = await data_table.toArray();
  tf.util.shuffle(points);

  //tf.split() gives Error: Number of splits must evenly divide the axis.
  // the length of the feature tensor should be even number.
  if(points.length %2 !== 0){
    points.pop();
  }

  const features_value = points.map(p=>p.x);
  const labels_value = points.map(p=>p.y);
  features_tensor = tf.tensor2d(features_value, [features_value.length,1]);
  labels_tensor = tf.tensor2d(labels_value, [labels_value.length,1]);
  // features_tensor.print();
  // labels_tensor.print();

  normalized_feature_tensor = normalize_tensor(features_tensor);
  normalized_label_tensor = normalize_tensor(labels_tensor);
  features_tensor.dispose();
  labels_tensor.dispose();

  [train_feature_tensor, test_feature_tensor] = tf.split(normalized_feature_tensor.tensor,2);
  [train_label_tensor, test_label_tensor] = tf.split(normalized_label_tensor.tensor,2);
  // train_feature_tensor.print();
  // train_label_tensor.print();

  console.log('# \tData loading -DONE-');
  return {
    normalized_feature_tensor,
    normalized_label_tensor,
    train_feature_tensor,
    test_feature_tensor,
    train_label_tensor,
    test_label_tensor
  };
}

async function train(model, train_feature_tensor, train_label_tensor){
  //console.log(train_feature_tensor.shape);
  //train_feature_tensor.print();
  const training_result = await train_model(model, train_feature_tensor,train_label_tensor);

  console.log(`#\tTraining set loss ${training_result.history.loss.pop().toPrecision(5)}`);
  console.log(`#\tValidation set loss ${training_result.history.val_loss.pop().toPrecision(5)}`);
  console.log('# \tTraining -DONE-');
}
// ======================================================================
// ======================================================================
async function run(){
  const data_obj = await load_data();
  const model = await create_model();
  console.log(model.summary());
  console.log('#==================== TRAIN MODEL ================================');
  console.log('#===============================================================');
  await train(model, data_obj.train_feature_tensor,data_obj.train_label_tensor );
  console.log(model.summary());
  console.log(model.getLayer(undefined, 0).getWeights().toString());

  console.log('#==================== TEST MODEL ================================');
  console.log('#===============================================================');
  await test_model(model, data_obj.test_feature_tensor, data_obj.test_label_tensor );
  console.log('# \tTest -DONE-\n');


  while (true) {
    const questions = [
      {
        type: 'number',
        name: 'house_sqft',
        message: "[Ctrl+c to end loop] \nEnter house size (sq. feet):"
      }
    ];
    const input_value = await inquirer.prompt(questions);
    //console.log(input_value.house_sqft);
    if(isNaN(input_value.house_sqft%2)){
      console.log('Enter a valid number');
    }
    else{
      tf.tidy(() => {
      const input_tensor = tf.tensor1d([input_value.house_sqft]);
      const normalized_input = normalize_tensor(input_tensor, data_obj.normalized_feature_tensor.min, data_obj.normalized_feature_tensor.max);
      // normalized_input.tensor.print();
      const prediction_output = model.predict(normalized_input.tensor);
      const denormal_prediction_output = denormalize_tensor(prediction_output, data_obj.normalized_label_tensor.min, data_obj.normalized_label_tensor.max );
      const prediction_output_value = denormal_prediction_output.dataSync()[0];
      console.log(`Predicted house price : \$ ${(prediction_output_value/1000).toFixed(0)*1000} `);
    });
    }
  }
}

run();
