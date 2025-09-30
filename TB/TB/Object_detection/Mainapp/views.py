import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from django.shortcuts import render
from django.http import JsonResponse
from django.conf import settings
from PIL import Image, ImageDraw, ImageFont
import torchvision.transforms as T
import os

MODEL_PATH = os.path.join(settings.BASE_DIR, "modal", "bacteria_model.pth")

def get_model(num_classes=2):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()
    return model

transform = T.Compose([T.ToTensor()])

def homepage(request):
    if request.method == "POST" and request.FILES.get("image"):
        try:
        
            model = get_model()

            img_file = request.FILES["image"]
            image = Image.open(img_file).convert("RGB")
            original_size = image.size
            img_tensor = transform(image).unsqueeze(0)

            with torch.no_grad():
                prediction = model(img_tensor)[0]

            
            result_image = image.copy()
            draw = ImageDraw.Draw(result_image)

        
            try:
                font = ImageFont.truetype("arial.ttf", 20)
            except:
                font = ImageFont.load_default()

            detection_count = 0
            total_confidence = 0
            detections = []

            for box, score in zip(prediction["boxes"], prediction["scores"]):
                if score > 0.5:  
                    detection_count += 1
                    total_confidence += score.item()
                    
                    x1, y1, x2, y2 = box.tolist()
                    
                    
                    draw.rectangle([x1, y1, x2, y2], outline="#00ff41", width=3)
                    
                
                    text = f"TB Bacteria: {score:.1%}"
                    bbox = draw.textbbox((x1, y1 - 25), text, font=font)
                    draw.rectangle([bbox[0]-2, bbox[1]-2, bbox[2]+2, bbox[3]+2], 
                                 fill=(0, 0, 0, 180), outline="#00ff41")
                    draw.text((x1, y1 - 25), text, fill="#00ff41", font=font)
                    
                    detections.append({
                        'box': [x1, y1, x2, y2],
                        'confidence': score.item(),
                        'label': 'TB Bacteria'
                    })

        
            result_path = os.path.join(settings.MEDIA_ROOT, "result.jpg")
            result_image.save(result_path, quality=95)

        
            avg_confidence = (total_confidence / detection_count) if detection_count > 0 else 0

            return JsonResponse({
                "result_image": settings.MEDIA_URL + "result.jpg",
                "detections": detections,
                "detection_count": detection_count,
                "avg_confidence": avg_confidence,
                "image_size": original_size,
                "status": "success",
                "message": f"Analysis completed. Found {detection_count} potential TB bacteria."
            })
            
        except Exception as e:
            return JsonResponse({
                "status": "error",
                "message": f"Error processing image: {str(e)}"
            }, status=500)

    return render(request, "index.html")
