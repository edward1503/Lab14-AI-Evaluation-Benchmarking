from __future__ import annotations

from typing import Dict, List


DOCS: List[Dict[str, object]] = [
    {
        "id": "IT-SEC-001",
        "title": "Password Reset Playbook",
        "topic": "account_security",
        "effective_date": "2026-01-10",
        "keywords": [
            "password",
            "mật khẩu",
            "reset",
            "đổi",
            "unlock",
            "self-service",
            "portal",
            "mfa",
            "employee id",
        ],
        "content": (
            "Nhân viên cần xác minh danh tính bằng MFA hoặc mã nhân viên trước khi đổi mật khẩu. "
            "Quy trình chuẩn là dùng Self-Service Password Portal. Nếu tài khoản đã bị khóa, "
            "nhân viên phải mở ticket cho IT Service Desk để được mở khóa trước khi đặt lại mật khẩu."
        ),
        "answer": (
            "Theo Password Reset Playbook, nhân viên phải xác minh bằng MFA hoặc mã nhân viên rồi dùng "
            "Self-Service Password Portal; nếu tài khoản đang bị khóa thì mở ticket cho IT Service Desk."
        ),
        "question_variants": [
            "Làm thế nào để đổi mật khẩu công ty khi vẫn đăng nhập được?",
            "Quy trình reset password nội bộ yêu cầu xác minh gì trước?",
            "Nếu tài khoản bị khóa thì bước tiếp theo để đặt lại mật khẩu là gì?",
        ],
    },
    {
        "id": "IT-SEC-002",
        "title": "MFA Enrollment Guide",
        "topic": "account_security",
        "effective_date": "2026-01-12",
        "keywords": [
            "mfa",
            "2fa",
            "xác thực",
            "authenticator",
            "sms",
            "backup code",
            "24 giờ",
        ],
        "content": (
            "Tài khoản mới phải kích hoạt MFA trong vòng 24 giờ kể từ lúc được cấp account. "
            "Ứng dụng Authenticator là phương thức mặc định. SMS chỉ được dùng làm phương án dự phòng "
            "cho nhân viên đang đi công tác và phải đăng ký cùng backup code."
        ),
        "answer": (
            "Theo MFA Enrollment Guide, nhân viên phải kích hoạt MFA trong 24 giờ, dùng ứng dụng "
            "Authenticator là mặc định; SMS chỉ là phương án dự phòng cho trường hợp đi công tác."
        ),
        "question_variants": [
            "Account mới phải bật MFA trong bao lâu?",
            "MFA mặc định của công ty là app hay SMS?",
            "Khi nào SMS được chấp nhận làm phương án xác thực dự phòng?",
        ],
    },
    {
        "id": "IT-NET-003",
        "title": "Production VPN Access Policy",
        "topic": "network_access",
        "effective_date": "2026-02-01",
        "keywords": [
            "vpn",
            "remote access",
            "thiết bị cá nhân",
            "managed device",
            "vdi",
            "1 ngày",
        ],
        "content": (
            "Chỉ thiết bị do công ty quản lý mới được truy cập production VPN. "
            "Thiết bị cá nhân không được kết nối trực tiếp mà phải dùng VDI đã được bảo vệ. "
            "Yêu cầu cấp VPN được xử lý trong 1 ngày làm việc sau khi quản lý phê duyệt."
        ),
        "answer": (
            "Production VPN chỉ dành cho thiết bị do công ty quản lý; thiết bị cá nhân phải dùng VDI, "
            "và yêu cầu cấp quyền được xử lý trong 1 ngày làm việc sau khi quản lý duyệt."
        ),
        "question_variants": [
            "Laptop cá nhân có được vào production VPN không?",
            "Yêu cầu cấp quyền VPN mất bao lâu sau khi manager approve?",
            "Nếu dùng máy cá nhân thì truy cập môi trường production bằng cách nào?",
        ],
    },
    {
        "id": "IT-HW-004",
        "title": "Laptop Replacement SLA",
        "topic": "it_support",
        "effective_date": "2026-01-15",
        "keywords": [
            "laptop",
            "replacement",
            "sla",
            "3 ngày",
            "manager approval",
            "mượn máy",
        ],
        "content": (
            "Laptop hỏng được thay trong 3 ngày làm việc sau khi có manager approval. "
            "Nếu máy hỏng nặng và ảnh hưởng công việc ngay, IT sẽ cấp máy mượn trong ngày. "
            "Người dùng phải sao lưu dữ liệu quan trọng lên ổ được quản lý trước khi bàn giao máy."
        ),
        "answer": (
            "Theo Laptop Replacement SLA, máy thay chính thức được cấp trong 3 ngày làm việc sau khi quản lý duyệt; "
            "nếu khẩn cấp thì IT cấp máy mượn trong ngày."
        ),
        "question_variants": [
            "Laptop hỏng thì bao lâu nhận máy thay?",
            "Trường hợp khẩn cấp có được mượn máy ngay không?",
            "Điều kiện để IT xử lý thay laptop theo SLA là gì?",
        ],
    },
    {
        "id": "HR-LEAVE-005",
        "title": "Annual Leave Submission Policy",
        "topic": "hr_policy",
        "effective_date": "2026-01-08",
        "keywords": [
            "nghỉ phép",
            "annual leave",
            "5 ngày",
            "hr portal",
            "3 ngày nghỉ",
            "manager approval",
        ],
        "content": (
            "Nhân viên xin nghỉ từ 3 ngày trở lên phải tạo yêu cầu trên HR Portal ít nhất 5 ngày làm việc trước ngày nghỉ. "
            "Quản lý trực tiếp là người phê duyệt đầu tiên. Nghỉ đột xuất dưới 3 ngày có thể tạo cùng ngày nhưng vẫn cần lý do rõ ràng."
        ),
        "answer": (
            "Theo Annual Leave Submission Policy, nghỉ từ 3 ngày trở lên phải nộp trên HR Portal trước ít nhất 5 ngày làm việc "
            "và cần quản lý trực tiếp phê duyệt."
        ),
        "question_variants": [
            "Muốn nghỉ phép 4 ngày thì cần gửi yêu cầu trước bao lâu?",
            "Xin annual leave dài ngày phải nộp ở đâu?",
            "Ai là người duyệt đầu tiên cho đơn nghỉ phép?",
        ],
    },
    {
        "id": "FIN-EXP-006",
        "title": "Expense Reimbursement Standard",
        "topic": "finance",
        "effective_date": "2026-01-20",
        "keywords": [
            "expense",
            "reimbursement",
            "hoàn ứng",
            "receipt",
            "500000",
            "7 ngày",
        ],
        "content": (
            "Mọi khoản chi trên 500.000 VND phải có hóa đơn hoặc biên nhận hợp lệ. "
            "Hồ sơ hoàn ứng được thanh toán trong vòng 7 ngày làm việc kể từ khi trưởng bộ phận và Finance cùng phê duyệt. "
            "Chi phí đi lại dưới 500.000 VND vẫn phải khai báo mục đích công tác."
        ),
        "answer": (
            "Expense Reimbursement Standard yêu cầu hóa đơn cho khoản chi trên 500.000 VND "
            "và hoàn ứng được xử lý trong 7 ngày làm việc sau khi đủ phê duyệt."
        ),
        "question_variants": [
            "Khoản chi công tác trên 500.000 có bắt buộc hóa đơn không?",
            "Finance hoàn ứng trong bao lâu sau khi hồ sơ được duyệt?",
            "Chi phí đi lại nhỏ hơn 500.000 có cần khai báo gì không?",
        ],
    },
    {
        "id": "SEC-INC-007",
        "title": "Phishing Incident Escalation",
        "topic": "security_incident",
        "effective_date": "2026-02-05",
        "keywords": [
            "phishing",
            "security incident",
            "15 phút",
            "hotline",
            "forward email",
            "sự cố",
        ],
        "content": (
            "Email nghi ngờ phishing phải được báo trong vòng 15 phút kể từ khi phát hiện. "
            "Người nhận phải forward thư đó đến hộp thư phishing@company.vn và đồng thời gọi security hotline nếu có dấu hiệu đã bấm vào link độc hại. "
            "Không được tự xóa log hoặc tự cô lập mailbox nếu chưa có hướng dẫn từ Security."
        ),
        "answer": (
            "Theo Phishing Incident Escalation, email nghi phishing phải được báo trong 15 phút, "
            "forward tới phishing@company.vn và gọi security hotline nếu đã lỡ tương tác với link độc hại."
        ),
        "question_variants": [
            "Khi nghi email phishing thì phải báo trong bao lâu?",
            "Email đáng ngờ cần forward tới đâu?",
            "Nếu đã bấm vào link phishing thì bước tiếp theo là gì?",
        ],
    },
    {
        "id": "OPS-DB-008",
        "title": "Production Database Recovery Objectives",
        "topic": "operations",
        "effective_date": "2026-02-09",
        "keywords": [
            "database",
            "backup",
            "recovery",
            "rpo",
            "rto",
            "15 phút",
            "4 giờ",
        ],
        "content": (
            "Hệ cơ sở dữ liệu production có Recovery Point Objective là 15 phút và Recovery Time Objective là 4 giờ. "
            "Bản backup full được tạo hằng đêm, còn incremental backup chạy mỗi 15 phút. "
            "Bài kiểm tra khôi phục phải được thực hiện hàng quý."
        ),
        "answer": (
            "Production Database Recovery Objectives quy định RPO là 15 phút và RTO là 4 giờ cho cơ sở dữ liệu production."
        ),
        "question_variants": [
            "RPO của database production là bao lâu?",
            "Mất tối đa bao lâu để khôi phục production DB theo mục tiêu hiện tại?",
            "Incremental backup của hệ DB chạy với chu kỳ nào?",
        ],
    },
    {
        "id": "DATA-PRIV-009",
        "title": "Customer Data Handling Standard",
        "topic": "privacy",
        "effective_date": "2026-01-18",
        "keywords": [
            "customer data",
            "dữ liệu khách hàng",
            "drive",
            "public link",
            "personal email",
            "90 ngày",
        ],
        "content": (
            "Dữ liệu khách hàng chỉ được chia sẻ qua thư mục công ty đã được cấp quyền hoặc hệ thống ticket nội bộ. "
            "Không gửi dữ liệu qua email cá nhân hoặc public link. "
            "Sau khi case đóng, dữ liệu tạm xuất phải xóa trong 90 ngày trừ khi có yêu cầu pháp lý."
        ),
        "answer": (
            "Customer Data Handling Standard yêu cầu chia sẻ dữ liệu khách hàng qua hệ thống nội bộ được cấp quyền, "
            "không dùng email cá nhân hay public link, và dữ liệu tạm xuất phải xóa trong 90 ngày sau khi case đóng."
        ),
        "question_variants": [
            "Có được gửi dữ liệu khách hàng qua email cá nhân không?",
            "Public link có được phép dùng để chia sẻ customer data không?",
            "Dữ liệu tạm xuất phải xóa sau bao lâu khi case đã đóng?",
        ],
    },
    {
        "id": "PROD-SUP-010",
        "title": "Support Severity Matrix",
        "topic": "customer_support",
        "effective_date": "2026-01-25",
        "keywords": [
            "p1",
            "p2",
            "severity",
            "response time",
            "15 phút",
            "1 giờ",
            "support",
        ],
        "content": (
            "Sự cố P1 phải được phản hồi lần đầu trong 15 phút và cập nhật mỗi 30 phút cho đến khi ổn định. "
            "Sự cố P2 có SLA phản hồi đầu tiên là 1 giờ. "
            "Mọi ticket P1 đều cần incident commander và bridge call."
        ),
        "answer": (
            "Support Severity Matrix quy định P1 phải phản hồi trong 15 phút, còn P2 là 1 giờ; P1 bắt buộc có incident commander."
        ),
        "question_variants": [
            "P1 cần phản hồi đầu tiên trong bao lâu?",
            "SLA phản hồi của ticket P2 là bao lâu?",
            "Ticket P1 có yêu cầu vai trò điều phối nào không?",
        ],
    },
    {
        "id": "HR-REMOTE-011",
        "title": "Remote Work Policy 2026",
        "topic": "hr_policy",
        "effective_date": "2026-01-15",
        "keywords": [
            "remote work",
            "work from home",
            "3 ngày",
            "anchor day",
            "thứ ba",
            "thứ tư",
            "thứ năm",
        ],
        "content": (
            "Từ ngày 15/01/2026, nhân viên được làm việc từ xa tối đa 3 ngày mỗi tuần nếu vai trò phù hợp. "
            "Các team phải có anchor days tại văn phòng vào thứ Ba đến thứ Năm. "
            "Mọi ngoại lệ vượt quá 3 ngày cần approval của Director."
        ),
        "answer": (
            "Theo Remote Work Policy 2026, nhân viên được làm remote tối đa 3 ngày mỗi tuần, "
            "còn thứ Ba đến thứ Năm là anchor days tại văn phòng nếu team không có ngoại lệ đã được duyệt."
        ),
        "question_variants": [
            "Hiện tại nhân viên được work from home tối đa mấy ngày mỗi tuần?",
            "Anchor days của chính sách remote hiện tại rơi vào ngày nào?",
            "Muốn remote quá 3 ngày thì cần ai phê duyệt?",
        ],
    },
    {
        "id": "HR-REMOTE-012",
        "title": "Legacy Remote Memo 2024",
        "topic": "hr_policy",
        "effective_date": "2024-07-01",
        "keywords": [
            "legacy",
            "2024",
            "remote",
            "2 ngày",
            "memo cũ",
            "superseded",
        ],
        "content": (
            "Bản memo năm 2024 từng cho phép tối đa 2 ngày remote mỗi tuần. "
            "Chính sách này đã bị thay thế bởi Remote Work Policy 2026 từ ngày 15/01/2026 và chỉ còn được lưu để đối chiếu lịch sử."
        ),
        "answer": (
            "Legacy Remote Memo 2024 chỉ còn giá trị tham chiếu lịch sử; chính sách hiện hành là Remote Work Policy 2026 với mức tối đa 3 ngày remote mỗi tuần."
        ),
        "question_variants": [
            "Memo remote năm 2024 quy định mấy ngày và còn hiệu lực không?",
            "Chính sách remote cũ khác gì bản 2026?",
            "Nếu thấy tài liệu 2024 ghi 2 ngày remote thì nên áp dụng tài liệu nào?",
        ],
    },
]


OUT_OF_CONTEXT_CASES: List[Dict[str, str]] = [
    {
        "question": "Công ty có chính sách hỗ trợ gửi xe máy theo tháng không?",
        "expected_answer": "Tôi chưa thấy tài liệu nào trong knowledge base hiện tại nói về chính sách gửi xe, nên chưa thể trả lời chắc chắn.",
    },
    {
        "question": "Thực đơn căng tin thứ Sáu có món gì?",
        "expected_answer": "Knowledge base hiện tại không có tài liệu về thực đơn căng tin, nên tôi chưa có thông tin xác thực để trả lời.",
    },
    {
        "question": "Bao giờ công ty mở bán cổ phiếu ESOP đợt tiếp theo?",
        "expected_answer": "Tôi không thấy tài liệu nào trong bộ hiện tại đề cập lịch ESOP tiếp theo, nên cần thêm nguồn chính thức.",
    },
    {
        "question": "Có chính sách nuôi thú cưng trong văn phòng không?",
        "expected_answer": "Tôi chưa tìm thấy tài liệu nội bộ nào về việc mang thú cưng vào văn phòng trong knowledge base này.",
    },
    {
        "question": "Mức phụ cấp ăn trưa hiện tại là bao nhiêu?",
        "expected_answer": "Knowledge base hiện tại không có chính sách phụ cấp ăn trưa, nên tôi chưa thể xác nhận con số chính xác.",
    },
    {
        "question": "Công ty đang dùng nhà cung cấp bảo hiểm nào cho thân nhân?",
        "expected_answer": "Tôi chưa thấy tài liệu nào trong bộ dữ liệu hiện tại nêu tên nhà cung cấp bảo hiểm cho thân nhân.",
    },
]


AMBIGUOUS_CASES: List[Dict[str, str]] = [
    {
        "question": "Thời hạn xử lý là bao lâu vậy?",
        "expected_answer": "Câu hỏi này chưa đủ ngữ cảnh vì nhiều quy trình có SLA khác nhau; bạn cần nói rõ đang hỏi về VPN, hoàn ứng, laptop hay ticket hỗ trợ.",
    },
    {
        "question": "Muốn làm từ xa thêm thì hỏi ai?",
        "expected_answer": "Bạn cần nói rõ đang hỏi ngoại lệ remote work nào; theo chính sách hiện tại, trường hợp vượt quá 3 ngày remote mỗi tuần cần Director phê duyệt.",
    },
    {
        "question": "Nếu có sự cố thì gửi vào đâu?",
        "expected_answer": "Bạn cần nêu rõ loại sự cố; riêng với email nghi phishing thì phải forward tới phishing@company.vn và có thể gọi security hotline.",
    },
    {
        "question": "Cái quy định cũ đó còn dùng không?",
        "expected_answer": "Bạn đang nói đến tài liệu nào; nếu là memo remote năm 2024 thì nó chỉ còn để tham chiếu lịch sử và đã bị chính sách 2026 thay thế.",
    },
    {
        "question": "Phải xác minh kiểu gì trước khi làm bước tiếp theo?",
        "expected_answer": "Câu hỏi này còn mơ hồ; nếu bạn đang nói tới reset mật khẩu thì cần xác minh bằng MFA hoặc mã nhân viên trước khi dùng portal.",
    },
    {
        "question": "Có cần hóa đơn không nhỉ?",
        "expected_answer": "Cần làm rõ bạn đang hỏi khoản chi nào; theo chính sách hoàn ứng, các khoản trên 500.000 VND bắt buộc có hóa đơn hợp lệ.",
    },
]


def get_documents() -> List[Dict[str, object]]:
    return [dict(doc) for doc in DOCS]
