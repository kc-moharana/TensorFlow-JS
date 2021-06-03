console.log('Creating Feature and label tensors');

console.log('Steps:');
console.log('1. read CSV data');
console.log('2. Export CSVDataset to Array');
console.log('3. Extract the desires features and labels as two separate tensors');


const tf = require('@tensorflow/tfjs');

async function run(){
  const file_location = 'file://C:/Users/suw169/Documents/GIT_repos/TensorFlow-JS/Exercise_codes/TFJS-node-server/data';

  const house_dataset = tf.data.csv(file_location+'/kc_house_data.csv');
  const sample_house_dataset = await house_dataset.take(10);
  const dataArray = sample_house_dataset.toArray();
  //console.log(dataArray);

  const points = house_dataset.map(record =>({
    x : record.sqft_living,
    y : record.price
  }) );
  //console.log(await points.toArray() );
  //plot_data(await points.toArray(), 'Square Feet');

 // Crete two tensors feature tensor and label tensor.
  const features_values = await points.map(p => p.x).toArray();
  const features_tensor = tf.tensor2d(features_values, [features_values.length,1]);

  const label_values = await points.map(p => p.y).toArray();
  const label_tensor = tf.tensor2d(label_values, [label_values.length,1]);

  features_tensor.print();
  label_tensor.print();
}

run();
