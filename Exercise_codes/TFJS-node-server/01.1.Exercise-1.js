console.log('Section2/Exercise-1');
const tf = require('@tensorflow/tfjs');


// created a tensor xs , and multiplied each of the values by 5
//  to create a new tensor ys with values
const xs = tf.tensor1d([1,2,3]);
const ys = xs.mul(tf.scalar(5));
ys.print();

// Create the tensor xs using the tf.tensor(...) function
const xs1 = tf.tensor([4.5,3.1,6.01,5.55,2.14,0.99]);

// Change it to a 2D tensor
xs1.reshape([3,2]).print();

//Try different operations such as .add(...) and .sub(...)
const a = tf.tensor1d([1, 2, 3, 4]);
const b = tf.tensor1d([3, 20, 30, 40]);

a.add(b).print();  // or tf.add(a, b)
a.sub(b).print();
b.sub(a).print();
a.sub(a).print();

// Try adding a 1D tensor to a 2D tensor.
const c = tf.tensor1d([9, 3, 3]);
//c.add(xs1.reshape([3,2])).print(); // Error: Operands could not be broadcast together with shapes 3 and 3,2.
c.add(xs1.reshape([2,3])).print(); // goves float dtype
