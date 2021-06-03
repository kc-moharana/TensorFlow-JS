console.log('Section2/Exercise-2');

const tf = require('@tensorflow/tfjs');

// y = mx + c
//   where m is a multiplier and c is a constant.
// One way to read it is that x is input and y is the output.

// Implement the method getYs(...) so that it returns a tensor
//   containing the y values for given x values, m, and c.
function getYs(xs, m, c) {
  // To implement
  const mx = tf.scalar(m).mul(xs);
  const y = tf.scalar(c).add(mx);
  return(y);
}
const t1 = tf.tensor1d([1,5,10]);
const t2 = getYs(t1, 2, 1);
t2.print();
