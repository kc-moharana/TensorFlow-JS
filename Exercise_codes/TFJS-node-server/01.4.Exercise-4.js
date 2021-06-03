
console.log('Section2/Exercise-4');
const tf = require('@tensorflow/tfjs');

for (let i = 0; i < 100; i++) {
    tf.tensor1d([1,2,3]);
}
for (let i = 0; i < 100; i++) {
    tf.tensor1d([4,5,6]);
}
console.log(tf.memory());

// Adapt the first loop using tf.dispose() to reduce memory usage.
for (let i = 0; i < 100; i++) {
   //tf.tensor1d([1,2,3]); //makes the numTensors: 300​
    tf.dispose(tf.tensor1d([1,2,3]) );//keeps the numTensors: 200​
}
console.log('tf.dispose() in Loop1');
console.log(tf.memory());
//Wrap the second loop in a function and use tf.tidy() to reduce memory usage.
tf.tidy(()=>{
  for (let i = 0; i < 100; i++) {
      tf.tensor1d([4,5,6]);
      console.log(i); // looping
  }
});
console.log(tf.memory());
