// Liste des labels (à adapter selon ton dataset)
const labels = [
    'pink primrose', 'hard-leaved pocket orchid', 'canterbury bells', 'sweet pea',
    'english marigold', 'tiger lily', 'moon orchid', 'bird of paradise', 'monkshood',
    'globe thistle', 'snapdragon', "colt's foot", 'king protea', 'spear thistle',
    'yellow iris', 'globe-flower', 'purple coneflower', 'peruvian lily',
    'balloon flower', 'giant white arum lily', 'fire lily', 'pincushion flower',
    'fritillary', 'red ginger', 'grape hyacinth', 'corn poppy',
    'prince of wales feathers', 'stemless gentian', 'artichoke', 'sweet william',
    'carnation', 'garden phlox', 'love in the mist', 'mexican aster',
    'alpine sea holly', 'ruby-lipped cattleya', 'cape flower', 'great masterwort',
    'siam tulip', 'lenten rose', 'barbeton daisy', 'daffodil', 'sword lily',
    'poinsettia', 'bolero deep blue', 'wallflower', 'marigold', 'buttercup',
    'oxeye daisy', 'common dandelion', 'petunia', 'wild pansy', 'primula',
    'sunflower', 'pelargonium', 'bishop of llandaff', 'gaura', 'geranium',
    'orange dahlia', 'pink-yellow dahlia?', 'cautleya spicata',
    'japanese anemone', 'black-eyed susan', 'silverbush', 'californian poppy',
    'osteospermum', 'spring crocus', 'bearded iris', 'windflower', 'tree poppy',
    'gazania', 'azalea', 'water lily', 'rose', 'thorn apple', 'morning glory',
    'passion flower', 'lotus', 'toad lily', 'anthurium', 'frangipani', 'clematis',
    'hibiscus', 'columbine', 'desert-rose', 'tree mallow', 'magnolia', 'cyclamen',
    'watercress', 'canna lily', 'hippeastrum', 'bee balm', 'ball moss', 'foxglove',
    'bougainvillea', 'camellia', 'mallow', 'mexican petunia', 'bromelia',
    'blanket flower', 'trumpet creeper', 'blackberry lily'
]

let session = null;

async function loadModel() {
    session = await ort.InferenceSession.create('flowers_102_net.onnx');
}

function preprocessImage(image, size = 224) {
    // Crée un canvas pour redimensionner l'image
    const canvas = document.createElement('canvas');
    canvas.width = size;
    canvas.height = size;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(image, 0, 0, size, size);
    // Récupère les données de pixels
    const imageData = ctx.getImageData(0, 0, size, size);
    const { data } = imageData;
    // Normalisation (à adapter selon ton entraînement)
    // Ici, on suppose [0,1] et pas de mean/std
    const input = new Float32Array(size * size * 3);
    for (let i = 0; i < size * size; i++) {
        input[i] = data[i * 4] / 255;     // R
        input[i + size * size] = data[i * 4 + 1] / 255; // G
        input[i + 2 * size * size] = data[i * 4 + 2] / 255; // B
    }
    // [1, 3, 224, 224]
    return new ort.Tensor('float32', input, [1, 3, size, size]);
}

function showImage(file) {
    const reader = new FileReader();
    reader.onload = function (e) {
        const img = document.createElement('img');
        img.src = e.target.result;
        img.onload = function () {
            document.getElementById('imagePreview').innerHTML = '';
            document.getElementById('imagePreview').appendChild(img);
        };
    };
    reader.readAsDataURL(file);
}

async function predict() {
    const preview = document.querySelector('#imagePreview img');
    if (!preview) return;
    document.getElementById('result').textContent = 'Analyse en cours...';
    const inputTensor = preprocessImage(preview);
    const feeds = { [session.inputNames[0]]: inputTensor };
    const output = await session.run(feeds);
    const outputTensor = output[session.outputNames[0]];
    const scores = outputTensor.data;
    // Trouver l'indice du score max
    const maxIdx = scores.indexOf(Math.max(...scores));
    const label = labels[maxIdx] || `Classe ${maxIdx}`;
    document.getElementById('result').textContent = `Résultat analyse : ${label}`;
}

document.getElementById('imageInput').addEventListener('change', function (e) {
    if (e.target.files && e.target.files[0]) {
        showImage(e.target.files[0]);
    }
});

document.getElementById('predictBtn').addEventListener('click', predict);

// Charger le modèle au démarrage
loadModel();
