console.log('Creating simple linear regression model using TFJS and training using .fit()');
console.log('Evaluate the model usig test data .evaluate');
console.log('split the 20% of the training set as validation data and train ');
console.log('Exercise1: add shuffle: true to model.fit() ');

const tf = require('@tensorflow/tfjs');

async function trainModel(model, trainFeature_tensor, trainLabelTensor){
   return model.fit(trainFeature_tensor,trainLabelTensor,
     {
      batchSize: 32,
       epochs:20,
       shuffle: true,
       callbacks:{
         onEpochEnd: (epoch, log)=>console.log(`Epoch ${epoch}: ${log.loss}`)
       }
     });
}


function createModel() {
  const model = tf.sequential();
  model.add(tf.layers.dense({
    units: 1,
    useBias: true,
    activation: 'linear',
    inputDim: 1,
  }));

  const optimizer = tf.train.sgd(0.1);
  model.compile({
    loss: 'meanSquaredError',
    optimizer,
  });
  return model;
}


// async function plot(datapoints, feature_name){
//   tfvis.render.scatterplot(
//   {name: 'Scatter plot '+`${feature_name} vs price`, tab:"Charts"},
//   {values:[datapoints], series:["Original"]},
//   {xLabel:"Square Feet", yLabel: "Price"}
//   );
// }
// scale a tensor 0-1
function normalize_tensor(xs){
  const min = xs.min();
  const max = xs.max();
  return {
    x: min.sub(xs).div(max.sub(min)),
    min: min,
    max: max
  }
}
function denormalize(tensor, min, max) {
  const denormalisedTensor = tensor.mul(max.sub(min)).add(min);
  return denormalisedTensor;
}

async function run(){
  await tf.ready();
  // Import CSV data
  const housingData = tf.data.csv('http://127.0.0.1:8080/data/kc_house_data.csv');
  const data_table = housingData.map(record =>({
    x: record.sqft_living,
    y: record.price
  }));

  const points = await data_table.toArray();
  tf.util.shuffle(points);
  // plot(points, "Square Feet");

  if(points.length %2 !== 0){
    points.pop();
  }

  const features_tensor = tf.tensor2d( points.map(p=>p.x), [points.length, 1] );
  const labels_tensor = tf.tensor2d( points.map(p=>p.y), [points.length, 1] );
  // features_tensor.print();
  // labels_tensor.print();
  const scaled_features = normalize_tensor(features_tensor);
  const scaled_labels = normalize_tensor(labels_tensor);

  const [train_feat, test_feat] = tf.split(scaled_features.x, 2);
  const [train_labels, test_labels] = tf.split(scaled_labels.x, 2);
  train_feat.print();
  train_labels.print();
  const model = createModel();
  console.log('');
  console.log('===================================================');
  // Trainign model
  const result = await trainModel(model,train_feat, train_labels);
  console.log('Training set loss:'+ result.history.loss.pop());
  // console.log('Validation set  loss:'+ result.history.val_loss.pop() );
  console.log('');
  // Evalutation using test data.
  const lossTensor = await model.evaluate(test_feat, test_labels);
  const loss = await lossTensor.dataSync();
  console.log('Test set loss:'+loss );
}

// ===========================================
run();
// const model = createModel();
// tfvis.show.modelSummary({ name: `Model Summary`, tab: `Model` }, model);
// tfvis.show.layer({ name: `Layer 1`, tab: `Model Inspection` }, model.getLayer(undefined, 0));
// tfvis.show.layer({ name: `Layer 2`, tab: `Model Inspection` }, model.getLayer(undefined, 1));
