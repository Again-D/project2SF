from fpdf import FPDF
import os

# ---- PDF 클래스 정의 ----
class PDF(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 16)
        self.cell(0, 10, 'PatchCore 사과 불량 검출 리포트', ln=True, align='C')
        self.ln(10)

    def section_title(self, title):
        self.set_font('Arial', 'B', 12)
        self.cell(0, 10, title, ln=True)
        self.ln(5)

    def section_body(self, body):
        self.set_font('Arial', '', 11)
        self.multi_cell(0, 10, body)
        self.ln(5)

# ---- 리포트 생성 ----
def generate_pdf(ok_count, ng_count, auto_threshold, output_dir='outputs'):
    pdf = PDF()
    pdf.add_page()

    # 1. 기본 통계
    pdf.section_title('1. 통계 요약')
    summary = f'''
- 정상 (OK): {ok_count}개
- 이상 (NG): {ng_count}개
- 자동 추천 Threshold: {auto_threshold:.3f}
    '''
    pdf.section_body(summary)

    # 2. Pie Chart 삽입
    if os.path.exists(os.path.join(output_dir, 'statistics_pie_chart.png')):
        pdf.section_title('2. 정상/이상 비율 (Pie Chart)')
        pdf.image(os.path.join(output_dir, 'statistics_pie_chart.png'), x=30, w=150)
        pdf.ln(10)

    # 3. Bar Chart 삽입
    if os.path.exists(os.path.join(output_dir, 'statistics_bar_chart.png')):
        pdf.section_title('3. 정상/이상 개수 (Bar Chart)')
        pdf.image(os.path.join(output_dir, 'statistics_bar_chart.png'), x=30, w=150)
        pdf.ln(10)

    # 4. 대표 이미지 샘플 삽입
    pdf.section_title('4. 대표 결과 이미지')
    sample_dir = os.path.join(output_dir, 'ok')
    samples = sorted(os.listdir(sample_dir))[:2]  # OK 샘플 2장
    sample_dir_ng = os.path.join(output_dir, 'ng')
    samples_ng = sorted(os.listdir(sample_dir_ng))[:2]  # NG 샘플 2장

    for img_file in samples + samples_ng:
        if os.path.exists(os.path.join(sample_dir, img_file)):
            pdf.image(os.path.join(sample_dir, img_file), w=90)
            pdf.ln(5)
        elif os.path.exists(os.path.join(sample_dir_ng, img_file)):
            pdf.image(os.path.join(sample_dir_ng, img_file), w=90)
            pdf.ln(5)

    # 5. PDF 저장
    pdf.output(os.path.join(output_dir, 'patchcore_report.pdf'))
    print("✅ PDF 리포트 생성 완료! 저장 위치:", os.path.join(output_dir, 'patchcore_report.pdf'))

# ---- 사용 예시 ----
generate_pdf(ok_count, ng_count, auto_threshold)
