
console.log('Creating training and test dataset from IRIS data');
console.log('Problem statement: Given Sepal and Petal lengths and width predict the class of Iris');

const tf = require('@tensorflow/tfjs');

//Convert the species names to spp codes.
const spp_map = {'setosa':1, 'versicolor':2,'virginica':3};

async function prepare_data(csv_url){
  const CSVData = tf.data.csv(csv_url);
  //console.log(await CSVData.take(5).toArray() );
  const data_table = CSVData.map(records =>({
    sepal_length: records['Sepal.Length'],
    sepal_width: records['Sepal.Width'],
    species: spp_map[records.Species]
  }));

  const points = await data_table.toArray();
  tf.util.shuffle(points);

  if(points.length %2 !== 0){
    points.pos();
    console.log('rows reduced to :'+ points.length);
  }
  //console.log(points);

  // const features_tensor = tf.tensor2d(
  //   points.map(p=>p.sepal_length).concat(points.map(p=>p.sepal_width)),
  //   [points.length,2]
  // );

  const features_tensor = tf.tensor2d(
   [points.map(p=>p.sepal_length), points.map(p=>p.sepal_width)]
 ).transpose();
  const labels_tensor = tf.tensor2d( points.map(p=>p.species), [points.length, 1] );
  features_tensor.print();
  labels_tensor.print();

  const [train_feat, test_feat] = tf.split(features_tensor,2);
  const [train_label, test_label] =tf.split(labels_tensor,2);
  // Error: Argument 'x' passed to 'split' must be numeric tensor, but got string tensor
  train_feat.print();
  train_label.print();
}



const file_location = 'file://C:/Users/suw169/Documents/GIT_repos/TensorFlow-JS/Exercise_codes/TFJS-node-server/data';
const final_data = prepare_data(file_location+'/iris_dataset.csv'));
