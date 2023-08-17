const formNode = document.querySelector("form");
const messageNode = document.querySelector("#message");
const chatList = document.querySelector("#chat-list-container");
let count = 1;
formNode.onsubmit = async e => {
  e.preventDefault();
  if (count) {
    const res = document.createElement("li");
    res.innerHTML = `<span>${messageNode.value}</span>`;
    res.className = "bg-green-600";
    await new Promise(resolve => setTimeout(resolve, 500));
    chatList.appendChild(res);
    count--;
    messageNode.value = "";
    return;
  }
  const response = await fetch("http://127.0.0.1:5000/get");
  const { message = "" } = await response.json();

  const req = document.createElement("li");
  req.innerHTML = `<span>${messageNode.value}</span>`;
  req.className = "bg-blue-600";
  chatList.appendChild(req);
  const res = document.createElement("li");
  res.innerHTML = `<span>${message}</span>`;
  res.className = "bg-green-600";
  await new Promise(resolve => setTimeout(resolve, 500));
  chatList.appendChild(res);
  messageNode.value = "";
};
