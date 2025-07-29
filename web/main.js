const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
let drawing = false;
ctx.fillStyle = "white";
ctx.fillRect(0, 0, canvas.width, canvas.height);

canvas.addEventListener('mousedown', () => (drawing = true));
canvas.addEventListener('mouseup', () => (drawing = false));
canvas.addEventListener('mousemove', draw);

function draw(e) {
  if (!drawing) return;
  ctx.fillStyle = 'black';
  ctx.beginPath();
  ctx.arc(e.offsetX, e.offsetY, 10, 0, Math.PI * 2);
  ctx.fill();
}

document.getElementById('clear').onclick = () => {
  ctx.fillStyle = "white";
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  document.getElementById('result').textContent = '';
};

document.getElementById('predict').onclick = async () => {
  const imageData = ctx.getImageData(0, 0, 280, 280);
  const input = preprocess(imageData);
  const session = await ort.InferenceSession.create('../model/mnist_model.onnx');
  const tensor = new ort.Tensor('float32', input, [1, 1, 28, 28]);
  const output = await session.run({ input: tensor });
  const predictions = output.output.data;
  const prediction = predictions.indexOf(Math.max(...predictions));
  document.getElementById('result').textContent = `Chiffre prédit : ${prediction}`;
};

function preprocess(imageData) {
  const data = imageData.data;
  const small = document.createElement('canvas');
  small.width = 28;
  small.height = 28;
  const smallCtx = small.getContext('2d');
  smallCtx.drawImage(canvas, 0, 0, 28, 28);
  const imgData = smallCtx.getImageData(0, 0, 28, 28);

  const gray = new Float32Array(28 * 28);
  for (let i = 0; i < 28 * 28; i++) {
    const r = imgData.data[i * 4];
    gray[i] = (255 - r) / 255; // Inversion pour correspondre à MNIST
  }
  return gray;
}