const tf = require('@tensorflow/tfjs');

console.log('Reading CSV data files');

const file_location = 'file://C:/Users/kcm/Documents/GIT_repos/TensorFlow-JS/Exercise_codes/TFJS-node-server/data';

// ###### Just writing this code will not work, need to put inside an async function and use awit ##
// const house_dataset = tf.data.csv(`${file_location}/kc_house_data.csv`);
// const numOfFeatures = (await house_dataset.columnNames()).length - 1;
// console.log('Number of Features: '+numOfFeatures);
// // const sample_house_dataset = await house_dataset.take(10);
// // const dataArray = sample_house_dataset.toArray();
// //console.log(dataArray);


async function run(){
  const house_dataset =  tf.data.csv(`${file_location}/kc_house_data.csv`);
  const numOfFeatures = (await house_dataset.columnNames()).length - 1;
  console.log('Number of Features: '+numOfFeatures);
  const sample_house_dataset =   house_dataset.take(10);
  const dataArray = await sample_house_dataset.toArray();
  // console.log(dataArray);

  // Two data columns.
  const data_points = dataArray.map(records =>({
    x: records.sqft_living,
    y: records.price
  }));
  console.log(data_points);
}

run();
