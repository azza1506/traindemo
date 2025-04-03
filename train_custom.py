import os
import torch
import evaluate
import numpy as np
import pandas as pd
import glob as glob
import torch.optim as optim
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
 
 
from PIL import Image
from zipfile import ZipFile
from tqdm import tqdm
from dataclasses import dataclass
from torch.utils.data import Dataset
from urllib.request import urlretrieve
from transformers import (
    VisionEncoderDecoderModel,
    TrOCRProcessor,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    default_data_collator
)

def seed_everything(seed_value):
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
 
seed_everything(42)
 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

@dataclass(frozen=True)
class TrainingConfig:
    BATCH_SIZE:    int = 16  # Reduce from 48 to 16 or 8
    EPOCHS:        int = 200
    LEARNING_RATE: float = 0.00005
 
@dataclass(frozen=True)
class DatasetConfig:
    DATA_ROOT:     str = 'dataset'
 
@dataclass(frozen=True)
class ModelConfig:
    MODEL_NAME: str = 'microsoft/trocr-base-printed'


def visualize(dataset_path):
    plt.figure(figsize=(15, 3))
    all_images = os.listdir(f"{dataset_path}/train/image")
    
    for i in range(15):
        plt.subplot(3, 5, i+1)
        image = plt.imread(f"{dataset_path}/train/image/{all_images[i]}")
        plt.imshow(image)
        plt.axis('off')
        plt.title(all_images[i].split('.')[0])
    plt.show()

visualize(DatasetConfig.DATA_ROOT)

# Dataframe for training
train_df = pd.read_json(os.path.join("dataset/train.json"))
train_df["image"] = train_df["image"].apply(lambda x: x.split("\\")[-1])
train_df
# # Dataframe for testing
test_df = pd.read_json(os.path.join("dataset/test.json"))
test_df["image"] = test_df["image"].apply(lambda x: x.split("\\")[-1])
test_df


# Augmentations.
train_transforms = transforms.Compose([
    transforms.ColorJitter(brightness=.5, hue=.3),
    transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),
])

class CustomOCRDataset(Dataset):
    """
    Custom dataset for loading images and text data
    """
    def __init__(self, root_dir, df, processor, max_target_length=128, transforms=None):
        self.root_dir = root_dir
        self.df = df
        self.processor = processor
        self.max_target_length = max_target_length
        self.transforms = transforms

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        file_name = self.df['image'][idx]
        text = self.df['text'][idx]
        image = Image.open(os.path.join(self.root_dir, file_name)).convert("RGB")

        if transforms: 
            image = self.transforms(image)

        pixel_values = self.processor(image, return_tensors="pt").pixel_values
        labels = self.processor.tokenizer(text, padding="max_length", max_length=self.max_target_length).input_ids
        labels = [label if label != self.processor.tokenizer.pad_token_id else -100 for label in labels]
        encoding = {
            "pixel_values": pixel_values.squeeze(),
            "labels": torch.tensor(labels)
        }
    
        return encoding

processor = TrOCRProcessor.from_pretrained(ModelConfig.MODEL_NAME)
train_dataset = CustomOCRDataset(
    root_dir=os.path.join(DatasetConfig.DATA_ROOT, 'train/image/'),
    df=train_df,
    processor=processor,
    transforms=train_transforms,
)
valid_dataset = CustomOCRDataset(
    root_dir=os.path.join(DatasetConfig.DATA_ROOT, 'test/image/'),
    df=test_df,
    processor=processor,
    transforms=train_transforms
)

model = VisionEncoderDecoderModel.from_pretrained(ModelConfig.MODEL_NAME)
model.to(device)
print(model)
# Total parameters and trainable parameters.
total_params = sum(p.numel() for p in model.parameters())
print(f"{total_params:,} total parameters.")
total_trainable_params = sum(
    p.numel() for p in model.parameters() if p.requires_grad)
print(f"{total_trainable_params:,} training parameters.")

# Set special tokens used for creating the decoder_input_ids from the labels.
model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
model.config.pad_token_id = processor.tokenizer.pad_token_id
# Set Correct vocab size.
model.config.vocab_size = model.config.decoder.vocab_size
model.config.eos_token_id = processor.tokenizer.sep_token_id
 
model.config.max_length = 64
model.config.early_stopping = True
model.config.no_repeat_ngram_size = 3
model.config.length_penalty = 2.0
model.config.num_beams = 4

optimizer = optim.AdamW(
    model.parameters(), lr=TrainingConfig.LEARNING_RATE, weight_decay=0.0005
)

cer_metric = evaluate.load('cer')
 
 
def compute_cer(pred):
    labels_ids = pred.label_ids
    pred_ids = pred.predictions
 
 
    pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
    labels_ids[labels_ids == -100] = processor.tokenizer.pad_token_id
    label_str = processor.batch_decode(labels_ids, skip_special_tokens=True)
 
 
    filtered_preds = []
    filtered_refs = []

    for pred, ref in zip(pred_str, label_str):
        if ref.strip():  # Chỉ giữ lại các câu tham chiếu không rỗng
            filtered_preds.append(pred)
            filtered_refs.append(ref)

    cer = cer_metric.compute(predictions=filtered_preds, references=filtered_refs)

 
 
    return {"cer": cer}

training_args = Seq2SeqTrainingArguments(
    predict_with_generate=True,
    eval_strategy='epoch',
    per_device_train_batch_size=TrainingConfig.BATCH_SIZE,
    per_device_eval_batch_size=TrainingConfig.BATCH_SIZE,
    fp16=True,
    output_dir='seq2seq_model_printed/',
    logging_strategy='epoch',
    save_strategy='epoch',
    save_total_limit=5,
    report_to='tensorboard',
    num_train_epochs=TrainingConfig.EPOCHS
)

class CustomSeq2SeqTrainer(Seq2SeqTrainer):
    def training_step(self, model, inputs, num_items_in_batch=None):
        # Remove num_items_in_batch from inputs
        if 'num_items_in_batch' in inputs:
            num_items = inputs.pop('num_items_in_batch')
        else:
            num_items = num_items_in_batch
            
        output = super().training_step(model, inputs)
        
        # Clear cache after each step
        torch.cuda.empty_cache()
        
        return output

# Initialize trainer.
trainer = CustomSeq2SeqTrainer(
    model=model,
    tokenizer=processor.image_processor,
    args=training_args,
    compute_metrics=compute_cer,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,
    data_collator=default_data_collator,
)

res = trainer.train()

processor = TrOCRProcessor.from_pretrained(ModelConfig.MODEL_NAME)
trained_model = VisionEncoderDecoderModel.from_pretrained('seq2seq_model_printed/checkpoint-'+str(res.global_step)).to(device)

def read_and_show(image_path):
    """
    :param image_path: String, path to the input image.
 
 
    Returns:
        image: PIL Image.
    """
    image = Image.open(image_path).convert('RGB')
    return image


def ocr(image, processor, model):
    """
    :param image: PIL Image.
    :param processor: Huggingface OCR processor.
    :param model: Huggingface OCR model.
 
 
    Returns:
        generated_text: the OCR'd text string.
    """
    # We can directly perform OCR on cropped images.
    pixel_values = processor(image, return_tensors='pt').pixel_values.to(device)
    generated_ids = model.generate(pixel_values)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return generated_text

def eval_new_data(
    data_path=os.path.join(DatasetConfig.DATA_ROOT, 'test/image', '*'),
    num_samples=50
):
    image_paths = glob.glob(data_path)
    for i, image_path in tqdm(enumerate(image_paths), total=len(image_paths)):
        if i == num_samples:
            break
        image = read_and_show(image_path)
        text = ocr(image, processor, trained_model)
        plt.figure(figsize=(7, 4))
        plt.imshow(image)
        plt.title(text)
        plt.axis('off')
        plt.show()
 
eval_new_data(
    data_path=os.path.join(DatasetConfig.DATA_ROOT, 'test/image', '*'),
    num_samples=100
)

# Đánh giá model bằng TextMatch, CER và WER
print("\n" + "="*50)
print("ĐÁNH GIÁ MODEL BẰNG TEXTMATCH, CER VÀ WER")
print("="*50)

def calculate_text_match(prediction, ground_truth):
    """
    Tính tỷ lệ trùng khớp hoàn toàn giữa dự đoán và ground truth
    """
    return 1 if prediction == ground_truth else 0

def calculate_wer(prediction, ground_truth):
    """
    Tính Word Error Rate giữa văn bản dự đoán và ground truth
    WER = (S + D + I) / N, với:
    - S: số từ bị thay thế
    - D: số từ bị xóa
    - I: số từ bị chèn thêm
    - N: số từ trong ground truth
    """
    import Levenshtein
    
    # Tách thành danh sách các từ
    pred_words = prediction.split()
    gt_words = ground_truth.split()
    
    # Tính khoảng cách Levenshtein giữa hai danh sách từ
    distance = Levenshtein.distance(pred_words, gt_words)
    
    # WER = (S + D + I) / N
    wer = distance / max(len(gt_words), 1)
    
    return wer

def evaluate_model_metrics():
    """
    Đánh giá model bằng TextMatch, CER và WER trên tập test
    """
    test_images = glob.glob(os.path.join(DatasetConfig.DATA_ROOT, 'test/image', '*'))
    
    # Tạo mapping từ tên file ảnh đến text thực tế
    ground_truth_dict = {}
    for idx, row in test_df.iterrows():
        ground_truth_dict[row['image']] = row['text']
    
    results = []
    cer_scores = []
    text_match_scores = []
    wer_scores = []  # Thêm danh sách để lưu các điểm WER
    
    print(f"Đang đánh giá trên {len(test_images)} ảnh...")
    
    for image_path in tqdm(test_images):
        image_filename = os.path.basename(image_path)
        if image_filename not in ground_truth_dict:
            continue
            
        ground_truth = ground_truth_dict[image_filename]
        image = read_and_show(image_path)
        prediction = ocr(image, processor, trained_model)
        
        # Tính các metrics
        cer_score = cer_metric.compute(predictions=[prediction], references=[ground_truth])
        text_match_score = calculate_text_match(prediction, ground_truth)
        wer_score = calculate_wer(prediction, ground_truth)  # Tính WER
        
        results.append({
            'image': image_filename,
            'prediction': prediction,
            'ground_truth': ground_truth,
            'cer': cer_score,
            'text_match': text_match_score,
            'wer': wer_score  # Thêm WER vào kết quả
        })
        
        cer_scores.append(cer_score)
        text_match_scores.append(text_match_score)
        wer_scores.append(wer_score)  # Thêm WER vào danh sách scores
    
    # Tính trung bình các metrics
    avg_cer = sum(cer_scores) / len(cer_scores)
    avg_text_match = sum(text_match_scores) / len(text_match_scores)
    avg_wer = sum(wer_scores) / len(wer_scores)  # Tính trung bình WER
    
    return results, avg_cer, avg_text_match, avg_wer

# Thực hiện đánh giá
results, avg_cer, avg_text_match, avg_wer = evaluate_model_metrics()  # Cập nhật để nhận thêm avg_wer

# Hiển thị kết quả tổng quan
print("\nKết quả đánh giá tổng quan:")
print(f"- Character Error Rate (CER) trung bình: {avg_cer:.4f}")
print(f"- Word Error Rate (WER) trung bình: {avg_wer:.4f}")  # Hiển thị WER trung bình
print(f"- Text Match Accuracy: {avg_text_match:.4f} ({int(avg_text_match * 100)}%)")

# Lưu kết quả chi tiết vào DataFrame và xuất ra CSV
results_df = pd.DataFrame(results)
csv_path = os.path.join("seq2seq_model_printed", "evaluation_results.csv")
results_df.to_csv(csv_path, index=False)
print(f"\nĐã lưu kết quả chi tiết vào: {csv_path}")

# Trực quan hóa phân phối CER
plt.figure(figsize=(10, 6))
plt.hist([r['cer'] for r in results], bins=20, alpha=0.7, color='blue')
plt.axvline(avg_cer, color='red', linestyle='dashed', linewidth=2, label=f'CER trung bình: {avg_cer:.4f}')
plt.title('Phân phối Character Error Rate (CER)')
plt.xlabel('CER')
plt.ylabel('Số lượng mẫu')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig(os.path.join("seq2seq_model_printed", "cer_distribution.png"))
plt.show()

# Trực quan hóa phân phối WER
plt.figure(figsize=(10, 6))
plt.hist([r['wer'] for r in results], bins=20, alpha=0.7, color='green')
plt.axvline(avg_wer, color='red', linestyle='dashed', linewidth=2, label=f'WER trung bình: {avg_wer:.4f}')
plt.title('Phân phối Word Error Rate (WER)')
plt.xlabel('WER')
plt.ylabel('Số lượng mẫu')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig(os.path.join("seq2seq_model_printed", "wer_distribution.png"))
plt.show()

# So sánh CER và WER
plt.figure(figsize=(10, 6))
plt.scatter([r['cer'] for r in results], [r['wer'] for r in results], alpha=0.6)
plt.title('Mối quan hệ giữa CER và WER')
plt.xlabel('Character Error Rate (CER)')
plt.ylabel('Word Error Rate (WER)')
plt.grid(True, alpha=0.3)
plt.savefig(os.path.join("seq2seq_model_printed", "cer_vs_wer.png"))
plt.show()

# Hiển thị một số ví dụ đúng và sai
def display_examples(results, category="good", num_examples=5):
    """
    Hiển thị các ví dụ dự đoán tốt hoặc xấu
    category: "good" hoặc "bad"
    """
    if category == "good":
        # Lấy các ví dụ có text_match = 1 (dự đoán đúng hoàn toàn)
        filtered_results = [r for r in results if r['text_match'] == 1]
        title = "Các ví dụ dự đoán đúng hoàn toàn"
    else:
        # Lấy các ví dụ có CER cao nhất (dự đoán sai nhiều)
        filtered_results = sorted(results, key=lambda x: x['cer'], reverse=True)
        title = "Các ví dụ dự đoán sai nhiều nhất"
    
    # Giới hạn số lượng ví dụ
    examples = filtered_results[:num_examples]
    
    plt.figure(figsize=(15, 4*num_examples))
    plt.suptitle(title, fontsize=16)
    
    for i, example in enumerate(examples):
        img_path = os.path.join(DatasetConfig.DATA_ROOT, 'test/image', example['image'])
        image = plt.imread(img_path)
        
        plt.subplot(num_examples, 1, i+1)
        plt.imshow(image)
        plt.axis('off')
        
        if category == "good":
            plt.title(f"Text: {example['prediction']}")
        else:
            plt.title(f"Dự đoán: '{example['prediction']}' | Thực tế: '{example['ground_truth']}' | CER: {example['cer']:.4f} | WER: {example['wer']:.4f}")
    
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    filename = "good_examples.png" if category == "good" else "bad_examples.png"
    plt.savefig(os.path.join("seq2seq_model_printed", filename))
    plt.show()

# Tạo confusion matrix cho các ký tự thường bị nhầm lẫn
def create_character_confusion(results, top_n=20):
    """
    Tạo và hiển thị các cặp ký tự thường bị nhầm lẫn nhất
    """
    char_confusions = {}
    
    for result in results:
        gt = result['ground_truth']
        pred = result['prediction']
        
        # Sử dụng thuật toán Levenshtein để tìm các ký tự bị nhầm lẫn
        import Levenshtein as lev
        
        # Tính toán ma trận operations
        ops = lev.editops(gt, pred)
        
        for op, gt_pos, pred_pos in ops:
            if op == 'replace':  # Chỉ quan tâm đến các thay thế
                gt_char = gt[gt_pos] if gt_pos < len(gt) else ''
                pred_char = pred[pred_pos] if pred_pos < len(pred) else ''
                
                if gt_char and pred_char:
                    pair = (gt_char, pred_char)
                    if pair in char_confusions:
                        char_confusions[pair] += 1
                    else:
                        char_confusions[pair] = 1
    
    # Sắp xếp các cặp ký tự theo tần suất
    sorted_confusions = sorted(char_confusions.items(), key=lambda x: x[1], reverse=True)
    
    # Lấy top_n cặp
    top_confusions = sorted_confusions[:top_n]
    
    if not top_confusions:
        print("Không tìm thấy đủ cặp ký tự bị nhầm lẫn")
        return
    
    # Hiển thị kết quả
    plt.figure(figsize=(12, 8))
    
    pairs = [f"'{gt}' → '{pred}'" for (gt, pred), _ in top_confusions]
    counts = [count for _, count in top_confusions]
    
    plt.barh(pairs, counts, color='salmon')
    plt.xlabel('Số lần xuất hiện')
    plt.ylabel('Cặp ký tự (Thực tế → Dự đoán)')
    plt.title(f'Top {len(top_confusions)} cặp ký tự thường bị nhầm lẫn')
    plt.tight_layout()
    plt.savefig(os.path.join("seq2seq_model_printed", "character_confusion.png"))
    plt.show()

# Thêm hàm phân tích nhầm lẫn ở cấp độ từ
def create_word_confusion(results, top_n=20):
    """
    Tạo và hiển thị các cặp từ thường bị nhầm lẫn nhất
    """
    word_confusions = {}
    
    for result in results:
        gt_words = result['ground_truth'].split()
        pred_words = result['prediction'].split()
        
        # Xác định các hoạt động thay thế, xóa, chèn
        import Levenshtein as lev
        
        # Tính toán ma trận operations giữa các từ
        ops = lev.editops(gt_words, pred_words)
        
        for op, gt_pos, pred_pos in ops:
            if op == 'replace':  # Chỉ quan tâm đến các thay thế
                gt_word = gt_words[gt_pos] if gt_pos < len(gt_words) else ''
                pred_word = pred_words[pred_pos] if pred_pos < len(pred_words) else ''
                
                if gt_word and pred_word:
                    pair = (gt_word, pred_word)
                    if pair in word_confusions:
                        word_confusions[pair] += 1
                    else:
                        word_confusions[pair] = 1
    
    # Sắp xếp các cặp từ theo tần suất
    sorted_confusions = sorted(word_confusions.items(), key=lambda x: x[1], reverse=True)
    
    # Lấy top_n cặp
    top_confusions = sorted_confusions[:top_n]
    
    if not top_confusions:
        print("Không tìm thấy đủ cặp từ bị nhầm lẫn")
        return
    
    # Hiển thị kết quả
    plt.figure(figsize=(12, 8))
    
    pairs = [f"'{gt}' → '{pred}'" for (gt, pred), _ in top_confusions]
    counts = [count for _, count in top_confusions]
    
    plt.barh(pairs, counts, color='lightgreen')
    plt.xlabel('Số lần xuất hiện')
    plt.ylabel('Cặp từ (Thực tế → Dự đoán)')
    plt.title(f'Top {len(top_confusions)} cặp từ thường bị nhầm lẫn')
    plt.tight_layout()
    plt.savefig(os.path.join("seq2seq_model_printed", "word_confusion.png"))
    plt.show()

# Phân tích ký tự bị nhầm lẫn
try:
    import Levenshtein
    print("\nPhân tích các ký tự thường bị nhầm lẫn:")
    create_character_confusion(results)
    
    print("\nPhân tích các từ thường bị nhầm lẫn:")
    create_word_confusion(results)
except ImportError:
    print("\nĐể phân tích ký tự và từ bị nhầm lẫn, hãy cài đặt thư viện Levenshtein: pip install python-Levenshtein")

# Phân tích mối quan hệ giữa độ dài văn bản và độ chính xác
plt.figure(figsize=(10, 6))
text_lengths = [len(r['ground_truth']) for r in results]
wers = [r['wer'] for r in results]

plt.scatter(text_lengths, wers, alpha=0.6, color='green')
plt.title('Mối quan hệ giữa độ dài văn bản và WER')
plt.xlabel('Độ dài văn bản (số ký tự)')
plt.ylabel('Word Error Rate (WER)')
plt.grid(True, alpha=0.3)
plt.savefig(os.path.join("seq2seq_model_printed", "text_length_vs_wer.png"))
plt.show()

# Phân tích mối quan hệ giữa số lượng từ và độ chính xác
plt.figure(figsize=(10, 6))
word_counts = [len(r['ground_truth'].split()) for r in results]
wers = [r['wer'] for r in results]

plt.scatter(word_counts, wers, alpha=0.6, color='purple')
plt.title('Mối quan hệ giữa số lượng từ và WER')
plt.xlabel('Số lượng từ')
plt.ylabel('Word Error Rate (WER)')
plt.grid(True, alpha=0.3)
plt.savefig(os.path.join("seq2seq_model_printed", "word_count_vs_wer.png"))
plt.show()

print("="*50)
print("HOÀN THÀNH ĐÁNH GIÁ MODEL")
print("="*50)
