import torch


def predict_image(image, model, transform, device, gender_names, age_group_names):
    """
    Predicts gender and age group for one PIL image.
    """
    model.eval()

    image = image.convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        gender_outputs, age_outputs = model(input_tensor)

        gender_probs = torch.softmax(gender_outputs, dim=1)
        age_probs = torch.softmax(age_outputs, dim=1)

        gender_confidence, gender_prediction = torch.max(gender_probs, dim=1)
        age_confidence, age_prediction = torch.max(age_probs, dim=1)

    gender_idx = int(gender_prediction.item())
    age_idx = int(age_prediction.item())

    result = {
        "predicted_gender": gender_names[gender_idx],
        "gender_confidence": float(gender_confidence.item()),
        "predicted_age_group": age_group_names[age_idx],
        "age_confidence": float(age_confidence.item())
    }

    return result
