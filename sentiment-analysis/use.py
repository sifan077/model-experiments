# 预测函数
import torch
from transformers import BertTokenizer, BertForSequenceClassification

# 原来的label_map
label_map = {'pos': 1, 'neg': 2, 'unsup': 3}

# 反转label_map，方便根据预测结果查找对应标签
id_to_label_map = {v: k for k, v in label_map.items()}


def predict(model, text):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()

    # 使用 BERT 分词器对输入文本进行编码
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    encoding = tokenizer(text, padding='max_length', truncation=True, max_length=512, return_tensors='pt')

    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)

    with torch.no_grad():
        # 模型前向传播，获取 logits
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        predicted_label_id = torch.argmax(outputs.logits, dim=-1).item()

    # 使用 id_to_label_map 将标签ID转换为可读标签
    return id_to_label_map[predicted_label_id]


if __name__ == '__main__':
    # 初始化BERT模型
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=3)
    # 从保存的模型中恢复参数
    model.load_state_dict(torch.load('saved_model/model_epoch_3.bin'))
    text = "So tell me - what serious boozer drinks Budweiser? How many suicidally-obsessed drinkers house a fully stocked and barely touched range of drinks in their lonely motel room that a millionaire playboy's bachelor-pad bar would be proud to boast? And what kind of an alcoholic tends to drink with the bottle held about 8 inches from his hungry mouth so that the contents generally spill all over his face? Not to mention wasting good whisky by dousing your girlfriend's tits with it, just so the cinema audience can get a good eyeful of Elisabeth Shue's assets.<br /><br />Cage seems to be portraying the most attention-seeking look-at-me alcoholic ever to have graced the screen while Shue looks more like a Berkely preppy slumming it for a summer than some seasoned street-walker. She is humiliated and subjugated as often as possible in this revolting movie with beatings, skin lacerations, anal rape and graphic verbal abuse - all of it completely implausible and included apparently only to convey a sense of her horribly demeaned state and offer the male viewers an astonishingly clichéd sentimental sexual fantasy of the 'tart-with-a-heart'.<br /><br />Still - I did watch it to the end, by which time I was actually laughing out loud as Shue's tough street hooker chopped carrots in the kitchen wanly, pathetically smiling while Cage - all eyes popping and shaking like like a man operating a road drill in an earthquake - grimaced and mugged his way through the final half-hour..."
    prediction = predict(model, text)
    print(prediction)
