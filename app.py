from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse
from os import listdir, path
import numpy as np
import  cv2, os, audio
import pickle
import uuid
import onnxruntime
onnxruntime.set_default_logger_severity(3)
import  io
from tqdm import tqdm


UPLOAD_DIR = "uploads"
OUTPUT_DIR = "results"

# os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

app = FastAPI()




def load_model(device="cpu"):
    model_path = "checkpoints/wav2lip_gan.onnx"
    session_options = onnxruntime.SessionOptions()
    session_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
    providers = ["CPUExecutionProvider"]

    session = onnxruntime.InferenceSession(model_path, sess_options=session_options, providers=providers)	
    
    return session
        
def get_smoothened_boxes(boxes, T):
    for i in range(len(boxes)):
        if i + T > len(boxes):
            window = boxes[len(boxes) - T:]
        else:
            window = boxes[i : i + T]
        boxes[i] = np.mean(window, axis=0)
    return boxes





def datagen(frames, mels,boxes):
    
    img_batch, mel_batch, frame_batch, coords_batch = [], [], [], []

    face_det_results = [[image[y1: y2, x1:x2], (y1, y2, x1, x2)] for image, (x1, y1, x2, y2) in zip(frames, boxes)]
    for i, m in enumerate(mels):
        idx = i%len(frames)
        frame_to_save = frames[idx].copy()
        face, coords = face_det_results[idx].copy()

        face = cv2.resize(face, (96,96))
            
        img_batch.append(face)
        mel_batch.append(m)
        frame_batch.append(frame_to_save)
        coords_batch.append(coords)

        if len(img_batch) >= 1:
            img_batch, mel_batch = np.asarray(img_batch), np.asarray(mel_batch)

            img_masked = img_batch.copy()
            img_masked[:, 96//2:] = 0
            
            img_batch = np.concatenate((img_masked, img_batch), axis=3) / 255.
            mel_batch = np.reshape(mel_batch, [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1])

            yield img_batch, mel_batch, frame_batch, coords_batch
            img_batch, mel_batch, frame_batch, coords_batch = [], [], [], []

    if len(img_batch) > 0:
        img_batch, mel_batch = np.asarray(img_batch), np.asarray(mel_batch)

        img_masked = img_batch.copy()
        img_masked[:, 96//2:] = 0

        img_batch = np.concatenate((img_masked, img_batch), axis=3) / 255.
        mel_batch = np.reshape(mel_batch, [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1])

        yield img_batch, mel_batch, frame_batch, coords_batch


# device = 'cpu'

def main(full_frames,boxes,audio_path,outfile_path):
	fps = 15
	mel_step_size = 16
	wav = audio.load_wav(io.BytesIO(audio_path), 16000)
	mel = audio.melspectrogram(wav)
 
	mel_chunks = []
	mel_idx_multiplier = 80./fps 
	i = 0
	while 1:
		start_idx = int(i * mel_idx_multiplier)
		if start_idx + mel_step_size > len(mel[0]):
			mel_chunks.append(mel[:, len(mel[0]) - mel_step_size:])
			break
		mel_chunks.append(mel[:, start_idx : start_idx + mel_step_size])
		i += 1
 
 
	full_frames = full_frames[:len(mel_chunks)]
 
	
	gen = datagen(full_frames.copy(), mel_chunks,boxes)
 
	for i, (img_batch, mel_batch, frames, coords) in enumerate(tqdm(gen, 
											total=int(np.ceil(float(len(mel_chunks))/1)))):
		if i == 0:
 
 
			frame_h, frame_w = full_frames[0].shape[:-1]
			out = cv2.VideoWriter(outfile_path, cv2.VideoWriter_fourcc(*'avc1'), fps, (frame_w, frame_h))
 
 
		
		img_batch = img_batch.transpose((0, 3, 1, 2)).astype(np.float32)
		mel_batch = mel_batch.transpose((0, 3, 1, 2)).astype(np.float32)		
 
		pred = model.run(None,{'mel_spectrogram':mel_batch, 'video_frames':img_batch})[0][0]
		pred = pred.transpose(1, 2, 0)*255
		pred = pred.astype(np.uint8)
		pred = pred.reshape((1, 96, 96, 3))		
 
		for p, f, c in zip(pred, frames, coords):
 
			y1, y2, x1, x2 = c
			p = cv2.resize(p, (x2 - x1, y2 - y1))
			f[y1:y2, x1:x2] = p
			out.write(f)
 
			
	out.release()
    # endd = time.time()
    # print('total time lippppppppppppppppppppppppppppppppppppppppp',endd-startt)

    
model = load_model("")
print ("Model loaded")

boxes = [[802, 227, 1088, 628], [801, 227, 1088, 626], [801, 227, 1088, 624], [801, 227, 1088, 624], [801, 227, 1088, 624], [801, 227, 1088, 624], [801, 227, 1088, 624], [802, 227, 1088, 624], [801, 226, 1088, 623], [801, 226, 1088, 623], [801, 226, 1087, 623], [800, 225, 1086, 624], [800, 225, 1087, 624], [800, 225, 1087, 624], [800, 225, 1086, 623], [800, 226, 1086, 623], [800, 226, 1086, 623], [800, 226, 1086, 623], [799, 227, 1085, 623], [799, 226, 1085, 623], [800, 225, 1086, 622], [800, 226, 1086, 622], [800, 226, 1086, 622], [799, 226, 1086, 622], [799, 225, 1085, 621], [800, 225, 1083, 624], [800, 225, 1083, 624], [797, 224, 1081, 623], [797, 224, 1081, 623], [797, 224, 1081, 623], [798, 224, 1082, 623], [797, 224, 1082, 622], [797, 225, 1081, 622], [797, 225, 1081, 622], [800, 226, 1083, 623], [799, 225, 1085, 622], [799, 225, 1085, 621], [799, 226, 1085, 622], [799, 226, 1085, 622], [799, 226, 1085, 622], [799, 226, 1085, 622], [799, 226, 1086, 623], [799, 226, 1086, 623], [799, 226, 1086, 623], [799, 226, 1087, 623], [800, 225, 1087, 623], [800, 225, 1087, 623], [800, 225, 1087, 623], [800, 225, 1087, 623], [800, 225, 1087, 623], [801, 225, 1088, 623], [801, 226, 1088, 623], [801, 227, 1088, 624], [802, 227, 1088, 623], [801, 227, 1088, 624], [801, 227, 1088, 624], [801, 227, 1088, 624], [801, 228, 1088, 624], [801, 228, 1088, 625], [801, 228, 1088, 625]]

boxes = np.array(boxes)

with open('half_frames.pkl', 'rb') as f:
    full_frames = pickle.load(f)

    
    

@app.post("/lipsync/")
async def lipsync(audio: UploadFile = File(...)):
    
    audio_filename = await audio.read()


    output_filename = os.path.join(OUTPUT_DIR, f"{uuid.uuid4()}_output.mp4")
    
    main(full_frames=full_frames,boxes=boxes,audio_path=audio_filename,outfile_path=output_filename)
    
    if not os.path.exists(output_filename):
        raise HTTPException(status_code=500, detail="Failed to generate output video")

    return {"video_url": f"http://localhost:8001/results/{os.path.basename(output_filename)}"}

@app.get("/results/{filename}")
async def get_output_video(filename: str):
    file_path = os.path.join(OUTPUT_DIR, filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(file_path, media_type='video/mp4')

if __name__ == '__main__':
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8001)
    
