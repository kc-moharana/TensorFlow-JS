
const tf = require('@tensorflow/tfjs');

const t3 = tf.tensor1d([25, 76, 4, 23, -5, 22]);
const max = t3.max(); // 76
const min = t3.min(); // -5

// dataSync (): Synchronously downloads the values from the tf.Tensor.
//              This blocks the UI thread until the values are ready, which can cause performance issues.
// const original = 23;
// const minAsNumber = min.dataSync()[0];
// const maxAsNumber = max.dataSync()[0];
// const normalised = (original - minAsNumber) / (maxAsNumber - minAsNumber);

function normalize(xs){
  const max = xs.max(); // 76
  const min = xs.min();
  const denominator = max.sub(min);
  const numerator = xs.sub(min);
  return numerator.div(denominator);
}
normalize(t3).print(); // [0.3703704, 1, 0.1111111, 0.345679, 0, 0.3333333] print.js:34:10
// expected: [0.3703704, 1, 0.1111111, 0.345679, 0, 0.3333333].
