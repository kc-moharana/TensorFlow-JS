
console.log('# TFJS: Tensor core concepts ');
const tf = require('@tensorflow/tfjs');

// tf.tensor (values, shape?, dtype?)
// - Creates a tf.Tensor with the provided values, shape and dtype.

// - Pass an array of values to create a vector. shape?, dtype? are inferred from input array.
const t = tf.tensor([22,33,12,8,90]);
t.print();

// tf.scalar (value, dtype?)
// - Creates rank-0 tf.Tensor (scalar) with the provided value and dtype.
const s = tf.scalar(34);
s.print();

const xs = tf.tensor1d([1,1,3,1,5,6,7]);
xs.print();

const ys = tf.tensor2d([2,1,45,10,25,16], [3,2] );
ys.print();

// tf.reshape (x, shape)
// - Reshapes a tf.Tensor to a given shape.
tf.reshape(ys,[2,3]).print();


// Change datatype
//
const flot_x = tf.tensor([1.1, 0.99, 3.21, 5.55]);
flot_x.print();
tf.cast(flot_x, 'int32').print();


// Generate a new tensor
//  tf.linspace (start, stop, num)
//  - Return an evenly spaced sequence of numbers over the given interval.

const m = tf.linspace(0,100, 20); // creates a tensor of length 20 between 0-100;
m.print();


// toString (verbose?)
// -Returns a human-readable description of the tensor. Useful for logging.
console.log(ys.toString(true));

console.log('=========SPLIT=========');
// Split dataset
//  tf.split (x, numOrSizeSplits, axis?)
const x = tf.tensor2d([1, 2, 3, 4, 5, 6, 7, 8], [2, 4]);
x.print();
console.log('numOrSizeSplits=2, axis=1');
const [a, b] = tf.split(x, 2, 1);
a.print();
b.print();

//console.log('numOrSizeSplits=2, axis=2'); // Error
//  All values in axis param must be in range [-2, 2) but got axis 2
console.log('numOrSizeSplits=2, axis=0');
const [o, p] = tf.split(x, 2, 0);
o.print();
p.print();

console.log('numOrSizeSplits=[1, 2, 1], axis=1');
const [c, d, e] = tf.split(x, [1, 2, 1], 1);
c.print();
d.print();
e.print();


// Transpose matrix
console.log('=========Transpose=========');
x.transpose().print();
