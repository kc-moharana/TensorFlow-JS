

const tf = require('@tensorflow/tfjs');
const tfvis = require('@tensorflow/tfjs-vis');
//import tfvis from "@tensorflow/tfjs-vis";


const data = [
{ index: 0, value: 50 },
{ index: 1, value: 100 },
{ index: 2, value: 150 },
];

// Get a surface
const surface = tfvis.visor().surface({ name: 'Barchart', tab: 'Charts' });

// Render a barchart on that surface
tfvis.render.barchart(surface, data, {});


// GIves error
