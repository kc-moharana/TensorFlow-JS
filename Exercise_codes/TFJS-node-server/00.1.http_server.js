const http = require('http');
const pt = require('path');
const fs = require('fs');

// Create server object
http.createServer((req, res) => {
// Write response
res.write('<h1>Hello World: http-server<\h1>');
res.end()
} )
.listen(5000,()=>console.log('Server started...\n\nOpen localhost:5000 on browser.\n Ctrl+C to stop'));
