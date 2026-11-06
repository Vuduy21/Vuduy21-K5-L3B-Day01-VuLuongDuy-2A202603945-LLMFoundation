# Báo cáo cá nhân — Day 01: Lọc tin nhắn rác bằng ML, DL, Embedding và LLM

> Đổi tên file này thành `MSSV_HoTen.md` của bạn, ví dụ `2A20260000_NguyenVanAn.md`, rồi điền từng mục. Xoá các dòng hướng dẫn in nghiêng khi xong.

- **MSSV:**
- **Họ và tên:**
- **SEED** (in ở ô đầu notebook):

## 1. Bằng chứng

| Tệp | Có? |
| --- | --- |
| [`day01_lab.ipynb`](day01_lab.ipynb) — notebook đã chạy, còn output | ☐ |
| [`lab1_report.json`](lab1_report.json) — do ô "Báo cáo cuối" sinh ra | ☐ |
| [`hand_calc.jpg`](hand_calc.jpg) — ảnh tính tay có các bước | ☐ |

## 2. Kết quả chính

                                 precision  recall  F1 (test)  F1 (SUB 40)  \
1. BoW + LogReg (tự code)            1.000   0.748      0.856        0.824   
2. BoW + MLP (deep learning)         1.000   0.820      0.901        0.788   
3a. Encoder (MiniLM) + kNN           0.852   0.937      0.893        0.974   
3b. Encoder (MiniLM) + LogReg        0.981   0.910      0.944        0.947   
3c. Encoder (MiniLM) + MLP head      1.000   0.919      0.958        0.947   
4a. LLM nhỏ zero-shot                1.000   1.000      1.000        1.000   
4b. LLM lớn zero-shot                1.000   0.950      0.974        0.974   
4c. LLM nhỏ few-shot                 1.000   0.950      0.974        0.974   

                                 train (s)  tin/s (cả lô)  p50 ms/tin  \
1. BoW + LogReg (tự code)            4.843      20410.869       0.048   
2. BoW + MLP (deep learning)         9.178       9361.246       0.239   
3a. Encoder (MiniLM) + kNN           2.792        282.009      11.308   
3b. Encoder (MiniLM) + LogReg        3.131       1244.103       6.667   
3c. Encoder (MiniLM) + MLP head      3.859       1574.398       6.798   
4a. LLM nhỏ zero-shot                0.000          1.381     675.629   
4b. LLM lớn zero-shot                0.000          0.863    1064.742   
4c. LLM nhỏ few-shot                 0.000          1.366     688.836   

                                 p95 ms/tin  USD / 1k tin  
1. BoW + LogReg (tự code)             0.083         0.000  
2. BoW + MLP (deep learning)          8.309         0.000  
3a. Encoder (MiniLM) + kNN           13.673         0.000  
3b. Encoder (MiniLM) + LogReg         8.119         0.000  
3c. Encoder (MiniLM) + MLP head       7.361         0.000  
4a. LLM nhỏ zero-shot              1093.320         0.013  
4b. LLM lớn zero-shot              1691.113         0.503  
4c. LLM nhỏ few-shot                925.478         0.025  

## 3. Dự đoán và thực tế

-predict("emb_tricky", "Encoder + LogReg đoán đúng nhiều hơn vì mô hình Embedding nắm bắt được ý nghĩa ngữ nghĩa của câu, nhận diện được các tin nhắn ngụy trang né từ khóa spam nhưng mang ý nghĩa quảng cáo/lừa đảo; trong khi mô hình BoW chỉ đếm từ khóa thô nên dễ bị đánh lừa.") ,trong khi kết quả BoW lại chính xác cao hơn,có thể là do data lakeage hoặc overfiting.
-predict("llm_models",
    "(1) Model lớn (gpt-6.1-terra) F1 cao hơn một chút, khoảng +0.02–0.05 "
    "(vd ~0.93 vs ~0.89).)
    Tuy nhiên ra kết quả model nhỏ lại ra kết quả cao hơn.Có thể do mô hình lớn hơn dễ sai hơn vì chúng quá tự tin vào khả năng ngôn ngữ nên thường "bịa đặt" ra câu trả lời nghe rất thuyết phục thay vì thừa nhận không biết. 
-predict("hard_eval", "Còn khoảng 80%,Nhóm Natural_spam khó nhất với BoW,Prompt_Injection khó nất với LLM")
Đã chính xác với Bow ,tuy nhiên sai với LLM.
## 4. Phân tích 50 case khó

                                F1 test thường (SUB)  F1 50 case khó  \
1. BoW + LogReg                                0.824           0.308   
2. BoW + MLP                                   0.788           0.341   
3a. Emb + kNN                                  0.974           0.667   
3b. Emb + LogReg                               0.947           0.630   
3c. Emb + MLP head                             0.947           0.615   
4a. LLM nhỏ zero-shot                          1.000           0.954   
4d. LLM nhỏ + prompt phòng thủ                 1.000           0.954   

                                acc 50 case khó  
1. BoW + LogReg                           0.526  
2. BoW + MLP                              0.526  
3a. Emb + kNN                             0.632  
3b. Emb + LogReg                          0.649  
3c. Emb + MLP head                        0.649  
4a. LLM nhỏ zero-shot                     0.947  
4d. LLM nhỏ + prompt phòng thủ            0.947  

•Nhóm LLM: Đạt điểm F1 tuyệt đối (1.000) ở tập test thường và giữ phong độ cực tốt ở các case khó (F1: 0.954, Acc: 0.947). Việc thêm prompt phòng thủ (4d) chưa tạo ra sự khác biệt so với zero-shot (4a), cho thấy mô hình nền tảng vốn đã rất mạnh và ổn định.
•Nhóm Embedding: Vượt trội rõ rệt so với BoW trên tập test thường (F1 > 0.94). Tuy nhiên, khi đối mặt với 50 case khó, hiệu suất bị sụt giảm mạnh (F1 tụt xuống khoảng 0.61 - 0.66), cho thấy các mô hình phân loại tuyến tính/MLP nông dựa trên vector embedding chưa bắt trọn được các ngữ cảnh phức tạp hoặc bị nhiễu.
•Nhóm 1 va 2: Đạt kết quả thấp nhất. Việc bỏ qua thứ tự từ khiến mô hình bất lực trước các case khó (F1 chỉ quanh mức 0.3), phản ánh rõ giới hạn của phương pháp trích xuất đặc trưng truyền thống.


## 5. Thử thách cá nhân

def build_classification_prompt(
    text: str,
    examples: list[tuple[str, str]] | None = None
) -> str:
    parts = [
        "You are an SMS spam filter. Your task is to classify an SMS message "
        'as either "spam" or "ham".',
        "",
        '- "spam": unsolicited or fraudulent messages such as advertisements, '
        "promotions, prize/lottery claims, phishing links, requests for money "
        "or personal information, premium-rate numbers.",
        '- "ham": normal, legitimate messages between people or genuine '
        "service notifications.",
        "",
        "Return your answer as a valid JSON object with exactly two fields:",
        '- "label": either "spam" or "ham".',
        '- "reason": a short explanation for your classification.',
        "",
        'Example output: {"label": "ham", "reason": "A normal personal message."}',
        "",
        "Do not add Markdown, code fences, or any text outside the JSON object.",
    ]

    if examples:
        parts += ["", "Examples:", ""]
        for ex_text, ex_label in examples:
            parts.append(
                f'Message: """{ex_text}"""\n'
                f'JSON: {{"label": "{ex_label.strip().lower()}", '
                f'"reason": "Classification based on the message content."}}\n'
            )
        parts.append("Now classify the following message.")

    parts += ["", f'Message: """{text}"""\nJSON:']

    return "\n".join(parts)
Thay vì chỉ gửi một dòng văn bản thô cho AI, hàm này đóng vai trò như một lớp trung gian để:
• Chuyển đổi dữ liệu thô thành câu lệnh chuẩn chuyên gia: Tự động đóng gói vai trò, định nghĩa luật (spam/ham), ví dụ minh họa (Few-shot) và tin nhắn cần kiểm tra vào một văn bản duy nhất.
• Biến LLM thành một hàm lập trình có cấu trúc: Ép AI phản hồi chính xác dưới dạng dữ liệu JSON sạch (không kèm văn bản thừa hoặc ký tự Markdown), giúp hệ thống phần mềm dễ dàng đọc hiểu và xử lý tự động ở các bước sau.
## 6. Phương án cho sản phẩm thật

Chọn phương án : Encoder (MiniLM) + MLP head,với F1 = 0,958,p95ms/tin =  7.361,p 50ms/tin =   6.798 .Đánh đổi sự chính xác tuyệt đối của mô hình LLM tuy nhiên độ trễ thấp,chi phí rẻ hơn.  
## 7. AI log và điều đã học
Tôi đã dùng AI để hỗ trợ viết các hàm build_classification_prompt, parse_label, call_openai, llm_classify_batch và để thảo luận các câu hỏi phân tích. Code do AI viết không phải lúc nào cũng chạy đúng ngay. Bản đầu của call_openai và llm_classify_batch không khớp với template: hàm trả về chuỗi thay vì cặp (text, latency) và thiếu tham số temperature/top_p. Khi dán code, tôi cũng làm sót lệnh return và import re, dẫn đến lỗi TypeError. Tôi đã gửi khung gốc và ô test cho AI để chỉnh lại, rồi chạy test cho đến khi đạt . Ở phần phân tích, dự đoán ban đầu của AI rằng LLM sẽ bị lấn át trên đường Pareto đã sai so với kết quả thực tế. Vì vậy tôi đối chiếu mọi nhận định với biểu đồ và bảng số liệu trước khi ghi vào bài. Qua đó, tôi rút ra rằng AI giúp làm nhanh hơn, nhưng mọi kết quả đều cần được kiểm chứng bằng test và dữ liệu thật.


