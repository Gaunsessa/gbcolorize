import * as ort from 'onnxruntime-web';
import { Jimp, ResizeStrategy } from 'jimp';

type Image = Awaited<ReturnType<typeof Jimp.read>>;

let model: Promise<ort.InferenceSession> = ort.InferenceSession.create("model.onnx");

const jimpToTensor = (image: Image) => {
    const buffer = new Float32Array(image.height * image.width * 3);
    
    for (let c = 0; c < 3; c++) 
        for (let y = 0; y < image.height; y++)
            for (let x = 0; x < image.width; x++)
                buffer[c * (image.height * image.width) + y * image.width + x] 
                    = (image.getPixelColor(x, y) >> ((3 - c) * 8) & 0xFF) / 255.0;
    
    return new ort.Tensor("float32", buffer, [1, 3, image.height, image.width]);
}

const cropImage = (image: Image) => {
    if ((image.width % 128 !== 0 || image.height % 112 !== 0) &&
        (image.width % 160 !== 0 || image.height % 144 !== 0))
        throw new Error("Image has invalid size!");
        
    const has_boarder = image.width % 160 === 0;
        
    // Scale
    const factor = has_boarder ? 160 / image.width : 128 / image.width;
    image = image.scale({ f: factor, mode: ResizeStrategy.NEAREST_NEIGHBOR }) as Image;
    
    // Remove border
    if (has_boarder)
        image = image.crop({ x: 16, y: 16, w: 128, h: 112 }) as Image;
        
    return image;
}

const processImageTensor = (imageTensor: ort.Tensor) => {
    const rgb = imageTensor.data as Float32Array;
    const buffer = new BigInt64Array(imageTensor.data.length / 3);
    
    const offset = imageTensor.dims[2]! * imageTensor.dims[3]!;
    
    for (let i = 0; i < buffer.length; i++)
        buffer[i] = BigInt(Math.round(rgb[i]! + rgb[i + offset]! + rgb[i + offset * 2]!));
    
    if (Array.from(new Set(buffer)).length !== 4) 
        throw new Error("Image has too many colors!");
    
    return new ort.Tensor("int64", buffer, [1, 1, imageTensor.dims[2]!, imageTensor.dims[3]!]);
}

const handleImage = async (event: Event) => {
    const input = event.target as HTMLInputElement | null;
    if (!input || !input.files || input.files.length === 0) return;

    const file = input.files[0]!;

    const arrayBuffer = await file.arrayBuffer();
    let image = await Jimp.read(arrayBuffer);
    
    image = cropImage(image);
    
    const imageTensor = jimpToTensor(image);
    
    const imageData = imageTensor.toImageData({ format: "RGB", tensorLayout: "NCHW" });
    
    const canvas = document.getElementById('inputCanvas') as HTMLCanvasElement;
    const ctx = canvas.getContext('2d')!;
    
    ctx.putImageData(imageData, 0, 0);
    
    const modelInput = processImageTensor(imageTensor);

    model.then(async (model) => {
        const pred = await model.run({ gb_image: modelInput });
        
        const canvas = document.getElementById('outputCanvas') as HTMLCanvasElement;
        const ctx = canvas.getContext('2d')!;
        
        ctx.putImageData(pred.color_image!.toImageData({ format: "RGB", tensorLayout: "NCHW" }), 0, 0);
    });
};

// Register handlers
document.querySelector('#imageInput')!.addEventListener('change', handleImage);
document.querySelector('#input')!.addEventListener('click', () => (document.querySelector('#imageInput')! as HTMLInputElement).click());