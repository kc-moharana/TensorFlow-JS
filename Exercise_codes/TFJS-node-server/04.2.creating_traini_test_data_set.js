console.log('Creaing training and test dataset');

console.log('Steps:');
console.log('1. read CSV data');
console.log('1.1 Shuffle the Dataset at this stage');
console.log('2. Export CSVDataset to Array');
console.log('2.1 Make sure the datatable has even number of rows; tfjs cannot handle odd number of rows');
console.log('3. Extract the desires features and labels as two separate tensors');
console.log('3.1 Normalize the dataset');
console.log('3.2 split the dataset into two evenly distributed dataset as trainign and test dataset');

const tf = require('@tensorflow/tfjs');

function normalize(xs){
  const max = xs.max(); // 76
  const min = xs.min();
  const denominator = max.sub(min);
  const numerator = xs.sub(min);
  return {
    x:numerator.div(denominator),
    min: min,
    max: max
  };
}


// ===========================================================
async function run(){
  const file_location = 'file://C:/Users/kcm/Documents/GIT_repos/TensorFlow-JS/Exercise_codes/TFJS-node-server/data';
  const house_dataset = tf.data.csv(file_location+'/kc_house_data.csv');
  // const sample_house_dataset = await house_dataset.take(10);
  // const dataArray = sample_house_dataset.toArray();
  //console.log(dataArray);
  const data_table = house_dataset.map(record =>({
    x : record.sqft_living,
    y : record.price
  }) );
 const points = await data_table.toArray();
  //tf.util.shuffle (array) : Shuffles the array in-place using Fisher-Yates algorithm.
  //const dataArray = await points.toArray();
  tf.util.shuffle(points);

  //console.log(points.length);
  if (points.length %2 !== 0){
    points.pop();
  }
  //console.log(points.length);

  //console.log(await points.toArray() );
  //plot_data(await points.toArray(), 'Square Feet');

 // Crete two tensors feature tensor and label tensor.
  const features_values = points.map(p => p.x);
  const features_tensor = tf.tensor2d(features_values, [features_values.length,1]);

  const label_values = points.map(p => p.y);
  const label_tensor = tf.tensor2d(label_values, [label_values.length,1]);

  // features_tensor.print();
  // label_tensor.print();

  //normalize
  const norm_features_tf = normalize(features_tensor);
  const norm_labels_tf = normalize(label_tensor);
  // norm_features_tf.x.print();
  // norm_labels_tf.x.print();

  // split into tes and training data
  const [training_features, test_features ] = tf.split(norm_features_tf.x, 2);
  const [training_labels, test_labels ] = tf.split(norm_labels_tf.x, 2);
  training_features.print();
  test_features.print();

  training_labels.print();
  test_labels.print();
}

run();
