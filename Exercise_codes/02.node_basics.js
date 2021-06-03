const url = require('url');
const myUrl = new URL('http://mywebsite.com:8080/hello.html?id=101&status=active');
console.log(myUrl.href);
// Root domain with port
console.log(myUrl.host);
// Domain name without port
console.log(myUrl.hostname);
// File name
console.log(myUrl.pathname);
// Parameter queries
console.log(myUrl.search);
// Parameters as an object
console.log(myUrl.searchParams);

// iterate over Parameters
myUrl.searchParams.forEach((item, i) => {
  console.log(i+' : '+ item);
});
