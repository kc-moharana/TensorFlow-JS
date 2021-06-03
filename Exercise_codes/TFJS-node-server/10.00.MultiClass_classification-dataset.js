console.log('Design a Commandline User Interface to accept a house_space and approx the house price and predict number of bedrooms in the house');
console.log('===========');

const inquirer = require('inquirer');
const tf = require('@tensorflow/tfjs');

// global variables (doesnt work in NodeJS, only in Browser mode)
// let model;
// let normalized_feature_tensor, normalized_label_tensor;
// let train_feature_tensor, test_feature_tensor, train_label_tensor, test_label_tensor;

// ======================================================================
function normalize_tensor(tensor, user_min=null, user_max=null){
  const featureDimensions = tensor.shape[1];
  if(featureDimensions > 1){
    // Multiple features
    const multiTensors = tf.split(tensor, featureDimensions,1 );
    const normalized_tensors = multiTensors.map( (i_tensor, i) =>
    normalize_tensor(i_tensor,
      user_min ? user_min[i]:null,
      user_max ? user_max[i]:null
    )
  );
    const returnTensor = tf.concat(normalized_tensors.map(f => f.tensor),1);
    const min = normalized_tensors.map(f => f.min);
    const max = normalized_tensors.map(f => f.max);
    return({tensor:returnTensor, min, max });
  }
  else{
    const min = user_min || tensor.min();
    const max = user_max || tensor.max();
    return {
      tensor: tensor.sub(min).div(max.sub(min)),
      min,
      max
    }
  }
}

function denormalize_tensor(tensor, min, max) {
  const featureDimensions = tensor.shape[1];
  if(featureDimensions > 1){
    // Multiple features
    const multiTensors = tf.split(tensor, featureDimensions,1 );
    const denormalized_tensors = multiTensors.map( (i_tensor, i) =>
    denormalize_tensor(i_tensor,
      min[i],
      max[i]
      )
    );
    const returnTensor = tf.concat(denormalized_tensors,1);
    return returnTensor;
  }
  else{
    const denormalisedTensor = tensor.mul(max.sub(min)).add(min);
    return denormalisedTensor;
  }
}

async function create_model(){
  const model = tf.sequential();
  model.add(tf.layers.dense({
    activation: 'sigmoid', // Activation function to use.
    useBias: true,        // Whether to apply a bias.
    units: 10,             // Positive integer, dimensionality of the output space.
    inputDim :2           //If specified, defines inputShape as [inputDim]. // two features
  }));
  model.add(tf.layers.dense({
    activation: 'sigmoid', // Activation function to use.
    useBias: true,        // Whether to apply a bias.
    units: 10,             // Positive integer, dimensionality of the output space.
  }));
  model.add(tf.layers.dense({
    activation: 'softmax', // Activation function to use. <====================== softmax to normalize output
    useBias: true,        // Whether to apply a bias.
    units: 3,             // Positive integer, dimensionality of the output space.<======== number of output classes
  }));
// const learningRate = 0.1;
// const optimizer = tf.train.sgd(learningRate);
const optimizer = tf.train.adam();


// Compiling outfits the model with an optimizer, loss, and/or metrics.
// Configures and prepares the model for training and evaluation.
model.compile({
  //loss: 'meanSquaredError', //The loss value that will be minimized by the model will then be the sum of all individual losses.
  loss: 'categoricalCrossentropy', // < ==========================
  optimizer
});
return model;
}

async function train_model(model,train_feature_tensor, train_label_tensor){
  const training_result = await model.fit(train_feature_tensor, train_label_tensor,
    {
      batchSize : 32,
      epochs : 100,
      shuffle : true,
      validationSplit: 0.2,
      callbacks:{
        onEpochEnd: (epoch, log)=>console.log(`Epoch ${epoch}: ${log.loss}`)
      }
    });

    console.log(`#\tTraining set loss ${training_result.history.loss.pop().toPrecision(5)}`);
    console.log(`#\tValidation set loss ${training_result.history.val_loss.pop().toPrecision(5)}`);
    console.log('# \tTraining -DONE-');
}

async function test_model(model, test_feature_tensor, test_label_tensor){
  const test_loss_output = model.evaluate(test_feature_tensor, test_label_tensor);
  const test_loss = await test_loss_output.dataSync()[0]
  console.log(`#\tTest set loss ${test_loss.toPrecision(5)}`);
}

// one-hot encoding of label tensors
function getClassIndex(className){
  if(className === 1 || className === "1" ){
    return 0;
  }
  else if (className === 2 || className === "2") {
    return 1;
  }
  else{
    return 2;
  }
}

function getClassName(classIndex){
  if (classIndex > 2) {
    return "3+";
  }
  else{
    return classIndex+1;
  }
}



async function load_data(){
  const csv_url = 'file://C:/Users/kcm/Documents/GIT_repos/TensorFlow-JS/Exercise_codes/TFJS-node-server/data/kc_house_data.csv';
  const CSVDataset = tf.data.csv(csv_url);
  //console.log(await CSVDataset.columnNames() );
  const data_table = CSVDataset.map(records=>({
    x: records.sqft_living ,
    y: records.price,
    class: records.bedrooms > 2? "3+" : records.bedrooms
  })).filter(r => r.class !== 0);
  
  //console.log(await data_table.take(10).toArray());
  const points = await data_table.toArray();
  tf.util.shuffle(points);

  //tf.split() gives Error: Number of splits must evenly divide the axis.
  // the length of the feature tensor should be even number.
  if(points.length %2 !== 0){
    points.pop();
  }

  const features_value = points.map(p=>[p.x, p.y]);
  const labels_value = points.map(p=> getClassIndex(p.class) );

  const features_tensor = tf.tensor2d(features_value);
  const labels_tensor = tf.tidy(()=> tf.oneHot(tf.tensor1d(labels_value,'int32'), 3) );  // tensor1d()

  // features_tensor.print();
  // labels_tensor.print();
  // [0, 0, 1],
  // [0, 0, 1],
  // [0, 0, 1],

  const normalized_feature_tensor = normalize_tensor(features_tensor);
  const normalized_label_tensor = normalize_tensor(labels_tensor);
  // // normalized_feature_tensor.tensor.print();
  // // normalized_label_tensor.tensor.print()
  //
  // // const denor_ft = denormalize_tensor(normalized_feature_tensor.tensor, normalized_feature_tensor.min, normalized_feature_tensor.max);
  // // const denor_lbl = denormalize_tensor(normalized_label_tensor.tensor, normalized_label_tensor.min, normalized_label_tensor.max);
  // // denor_ft.print();
  // // denor_lbl.print();
  //
  features_tensor.dispose();
  labels_tensor.dispose();

  [train_feature_tensor, test_feature_tensor] = tf.split(normalized_feature_tensor.tensor,2);
  [train_label_tensor, test_label_tensor] = tf.split(normalized_label_tensor.tensor,2);
  train_feature_tensor.print();
  test_label_tensor.print();

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

// ======================================================================
// ======================================================================
async function run(){
  const data_obj = await load_data();
  const model = await create_model();
  console.log(model.summary());
  console.log('#==================== TRAIN MODEL ================================');
  console.log('#===============================================================');
  await train_model(model, data_obj.train_feature_tensor,data_obj.train_label_tensor );
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
      },
      {
        type: 'number',
        name: 'price',
        message: "Enter house price (dollar):"
      }
    ];
    const input_value = await inquirer.prompt(questions);
    //console.log(input_value.house_sqft);
    if(isNaN(input_value.house_sqft%2) || isNaN(input_value.price%2) ){
      console.log('Enter a valid number');
    }
    else{
      tf.tidy(() => {
      const input_tensor = tf.tensor2d([[input_value.house_sqft, input_value.price ]]);
      const normalized_input = normalize_tensor(input_tensor, data_obj.normalized_feature_tensor.min, data_obj.normalized_feature_tensor.max);
      // normalized_input.tensor.print();
      const prediction_output = model.predict(normalized_input.tensor);
      const denormal_prediction_output = denormalize_tensor(prediction_output, data_obj.normalized_label_tensor.min, data_obj.normalized_label_tensor.max );
      const prediction_output_value = denormal_prediction_output.dataSync();

      let stringOutput ="";
      for (var i = 0; i < 3; i++) {
        stringOutput +=`\tLikelihood of getting ${getClassName(i)} bedrooms is : ${(prediction_output_value[i]*100).toFixed(1)}%\n`
      }
      console.log(stringOutput+'\n\n');
    });
    }
  }
}

run();
