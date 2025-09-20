let canvas = document.getElementById('canvas');
canvas.width = 400;   // broaden canvas
canvas.height = 400;  // broaden canvas
let ctx = canvas.getContext('2d');
let drawing = false;

// Pen settings (sharper look)
ctx.lineWidth = 1.5;  // slightly thinner
ctx.lineJoin = 'round';
ctx.lineCap = 'round';
ctx.strokeStyle = 'black';

canvas.addEventListener('mousedown', () => drawing = true);
canvas.addEventListener('mouseup', () => {
    drawing = false;
    ctx.beginPath();
});
canvas.addEventListener('mousemove', draw);

function draw(e) {
    if (!drawing) return;
    ctx.lineTo(e.offsetX, e.offsetY);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(e.offsetX, e.offsetY);
}

function clearCanvas() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.beginPath();
}

async function sendImage() {
    let dataURL = canvas.toDataURL("image/png");
    let response = await fetch("/predict", {
        method: "POST",
        body: JSON.stringify({ image: dataURL }),
        headers: { "Content-Type": "application/json" }
    });
    let result = await response.json();
    document.getElementById("result").innerText = "Prediction: " + (result.prediction || result.error);
}
