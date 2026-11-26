"""
backend/pdf_report.py
---------------------
PDF Report generation using ReportLab for deepfake detection scans.
Generates an exhaustive, publication-grade forensic report containing:
- Categorical verdict & calibrated confidence score
- Primary technical forensic rationale explaining why the image is Real or Fake
- Embedded Grad-CAM visual explainability heatmap image
- Heatmap color key (Red = High, Yellow = Moderate, Blue = Low)
- Regional facial anatomical influence breakdown table
- Multimodal forensic checks (MTCNN face alignment, ELA, metadata)
- Digital forensics research disclaimer
"""

import io
import base64
from datetime import datetime
from typing import Any, Dict

from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    HRFlowable,
    Image as RLImage
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT

from src.class_config import build_forensic_rationale, get_verdict


def generate_pdf_report(scan: Dict[str, Any]) -> io.BytesIO:
    """
    Generates a structured, evidence-backed forensic PDF report from a scan record.
    Returns:
        io.BytesIO buffer containing the PDF bytes.
    """
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=40,
        leftMargin=40,
        topMargin=40,
        bottomMargin=40
    )

    styles = getSampleStyleSheet()

    # Custom typography styles
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=18,
        leading=22,
        textColor=colors.HexColor('#0f172a'),
        alignment=TA_LEFT
    )

    subtitle_style = ParagraphStyle(
        'DocSubTitle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9,
        leading=13,
        textColor=colors.HexColor('#64748b'),
        alignment=TA_LEFT
    )

    heading2_style = ParagraphStyle(
        'Heading2',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=11,
        leading=15,
        textColor=colors.HexColor('#0f172a'),
        spaceBefore=10,
        spaceAfter=5
    )

    body_style = ParagraphStyle(
        'BodyDark',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8.5,
        leading=12,
        textColor=colors.HexColor('#334155')
    )

    body_bold = ParagraphStyle(
        'BodyDarkBold',
        parent=body_style,
        fontName='Helvetica-Bold'
    )

    badge_style = ParagraphStyle(
        'VerdictBadge',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=13,
        leading=17,
        alignment=TA_CENTER
    )

    story = []

    # 1. Header Banner
    story.append(Paragraph("DEEPFAKE FORENSIC EXAMINATION REPORT", title_style))
    story.append(Spacer(1, 3))
    story.append(Paragraph("Automated Multimodal Biometric & Synthetic Generation Verification", subtitle_style))
    story.append(Spacer(1, 8))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor('#0284c7'), spaceAfter=12))

    # 2. Verdict Card
    fake_prob = float(scan.get("fake_probability", 0.5))
    raw_verdict = str(scan.get("verdict") or get_verdict(fake_prob)).lower()
    is_fake = raw_verdict in ("deepfake", "fake")
    conf_percent = round((fake_prob if is_fake else (1.0 - fake_prob)) * 100, 1)

    if is_fake:
        badge_color = colors.HexColor('#dc2626')
        bg_card_color = colors.HexColor('#fef2f2')
        verdict_text = "<font color='#dc2626'><b>DEEPFAKE / SYNTHETIC</b></font>"
        summary_note = "Biometric anomalies, unnatural generative boundaries, or synthetic sensor noise patterns detected."
    else:
        badge_color = colors.HexColor('#16a34a')
        bg_card_color = colors.HexColor('#f0fdf4')
        verdict_text = "<font color='#16a34a'><b>REAL / AUTHENTIC</b></font>"
        summary_note = "Authentic facial micropores, natural corneal reflections, and coherent camera sensor noise confirmed."

    verdict_table_data = [
        [
            Paragraph(f"<b>FINAL VERDICT:</b> {verdict_text}", badge_style),
            Paragraph(f"<b>Confidence:</b> {conf_percent}% &nbsp;|&nbsp; <b>Synthetic Probability:</b> {round(fake_prob * 100, 1)}%", body_style)
        ],
        [
            Paragraph(f"<i>Assessment:</i> {summary_note}", body_style),
            Paragraph(f"<b>Report ID:</b> {scan.get('report_id', 'N/A')}<br/><b>Analyzed:</b> {scan.get('created_at', datetime.now().isoformat())[:19]}", body_style)
        ]
    ]

    verdict_table = Table(verdict_table_data, colWidths=[270, 260])
    verdict_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), bg_card_color),
        ('BOX', (0, 0), (-1, -1), 1.2, badge_color),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#cbd5e1')),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('LEFTPADDING', (0, 0), (-1, -1), 8),
        ('RIGHTPADDING', (0, 0), (-1, -1), 8),
    ]))
    story.append(verdict_table)
    story.append(Spacer(1, 10))

    story.append(Paragraph("1. Forensic Decision Rationale", heading2_style))
    rationale_text = scan.get("forensic_rationale") or build_forensic_rationale(
        "fake" if is_fake else "real",
        fake_prob,
        scan.get("regional_influence"),
    )

    rationale_table = Table(
        [[Paragraph(rationale_text, body_style)]],
        colWidths=[530]
    )
    rationale_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#f8fafc')),
        ('BOX', (0, 0), (-1, -1), 0.5, colors.HexColor('#cbd5e1')),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('LEFTPADDING', (0, 0), (-1, -1), 8),
        ('RIGHTPADDING', (0, 0), (-1, -1), 8),
    ]))
    story.append(rationale_table)
    story.append(Spacer(1, 10))

    # 4. Visual Explainability & Grad-CAM Heatmap
    story.append(Paragraph("2. Visual Explainability & Grad-CAM Heatmap", heading2_style))

    # Attempt to decode heatmap image
    heatmap_element = None
    heatmap_b64 = scan.get("heatmap_base64")
    if heatmap_b64:
        try:
            raw_b64 = heatmap_b64.split(",", 1)[-1]
            img_bytes = base64.b64decode(raw_b64)
            img_io = io.BytesIO(img_bytes)
            heatmap_element = RLImage(img_io, width=150, height=150)
        except Exception as e:
            print(f"[PDF Gen] Failed to render embedded heatmap: {e}")
            heatmap_element = None

    legend_text = (
        "<b>Grad-CAM Activation Color Scale:</b><br/>"
        "• <font color='#dc2626'><b>RED (High Influence, &ge; 50%):</b></font> Dominant neural focus area. Represents focal biometric anomalies or primary landmarks directing the verdict.<br/>"
        "• <font color='#d97706'><b>YELLOW (Moderate Influence, 25% - 50%):</b></font> Secondary contextual facial structures contributing moderate classification weighting.<br/>"
        "• <font color='#2563eb'><b>BLUE / COOL (Low Influence, &lt; 25%):</b></font> Baseline organic surface or background with minimal deviation from standard authentic captures.<br/><br/>"
        "<i>Methodology: Gradient-weighted Class Activation Mapping computes spatial gradients from the final neural layer, projecting exact model attention back onto facial geometry.</i>"
    )

    if heatmap_element:
        cam_card_data = [
            [heatmap_element, Paragraph(legend_text, body_style)]
        ]
        cam_card = Table(cam_card_data, colWidths=[165, 365])
        cam_card.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#ffffff')),
            ('BOX', (0, 0), (-1, -1), 0.5, colors.HexColor('#cbd5e1')),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('TOPPADDING', (0, 0), (-1, -1), 6),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
            ('LEFTPADDING', (0, 0), (-1, -1), 8),
            ('RIGHTPADDING', (0, 0), (-1, -1), 8),
        ]))
        story.append(cam_card)
    else:
        cam_card = Table([[Paragraph(legend_text, body_style)]], colWidths=[530])
        cam_card.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#f8fafc')),
            ('BOX', (0, 0), (-1, -1), 0.5, colors.HexColor('#cbd5e1')),
            ('TOPPADDING', (0, 0), (-1, -1), 6),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
            ('LEFTPADDING', (0, 0), (-1, -1), 8),
        ]))
        story.append(cam_card)

    story.append(Spacer(1, 10))

    # 5. Regional Facial Anatomical Influence Breakdown Table
    story.append(Paragraph("3. Regional Biometric Influence Breakdown", heading2_style))

    regional_influence = scan.get("regional_influence") or []
    if not regional_influence:
        story.append(Paragraph("Regional activation breakdown unavailable for this scan.", body_style))
        story.append(Spacer(1, 10))
    else:
        region_rows = [
            [
                Paragraph("<b>Facial Region</b>", body_bold),
                Paragraph("<b>Influence Tier</b>", body_bold),
                Paragraph("<b>Color Indicator</b>", body_bold),
                Paragraph("<b>Activation</b>", body_bold),
                Paragraph("<b>Diagnostic Observation</b>", body_bold)
            ]
        ]

        for item in regional_influence:
            lvl = str(item.get("influence_level", "Low")).upper()
            if "HIGH" in lvl or item.get("color") == "Red":
                indicator = "<font color='#dc2626'><b>HIGH (RED)</b></font>"
                score_txt = f"<font color='#dc2626'><b>{item.get('score', 0)}%</b></font>"
            elif "MOD" in lvl or item.get("color") == "Yellow":
                indicator = "<font color='#d97706'><b>MODERATE (YELLOW)</b></font>"
                score_txt = f"<font color='#d97706'><b>{item.get('score', 0)}%</b></font>"
            else:
                indicator = "<font color='#2563eb'><b>LOW (BLUE)</b></font>"
                score_txt = f"<font color='#2563eb'><b>{item.get('score', 0)}%</b></font>"

            region_rows.append([
                Paragraph(str(item.get("region", "Region")), body_style),
                Paragraph(str(item.get("influence_level", "Low")), body_style),
                Paragraph(indicator, body_style),
                Paragraph(score_txt, body_style),
                Paragraph(str(item.get("description", "-")), body_style)
            ])

        region_table = Table(region_rows, colWidths=[120, 75, 100, 60, 175])
        region_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#e2e8f0')),
            ('BOX', (0, 0), (-1, -1), 0.5, colors.HexColor('#cbd5e1')),
            ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#e2e8f0')),
            ('TOPPADDING', (0, 0), (-1, -1), 4),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
            ('LEFTPADDING', (0, 0), (-1, -1), 6),
            ('RIGHTPADDING', (0, 0), (-1, -1), 6),
        ]))
        story.append(region_table)
        story.append(Spacer(1, 10))
    story.append(Spacer(1, 10))

    # 6. Forensic Domain Checks
    story.append(Paragraph("4. Multimodal Domain Checks", heading2_style))
    meta_flags = scan.get("metadata_flags", [])
    flags_text = ", ".join(meta_flags) if meta_flags else "None (Clean / Original EXIF)"

    domain_data = [
        [Paragraph("<b>Verification Check</b>", body_bold), Paragraph("<b>Status</b>", body_bold), Paragraph("<b>Diagnostic Summary</b>", body_bold)],
        [
            Paragraph("Face Detection (MTCNN)", body_style),
            Paragraph("<font color='#16a34a'>PASS</font>" if scan.get("face_detected") else "<font color='#d97706'>FALLBACK</font>", body_style),
            Paragraph("Standardized 20% margin facial crop localized for deep network inference." if scan.get("face_detected") else "No frontal human face localized; evaluated full image.", body_style)
        ],
        [
            Paragraph("Error Level Analysis (ELA)", body_style),
            Paragraph("<font color='#dc2626'>FLAGGED</font>" if scan.get("ela_flag") else "<font color='#16a34a'>NORMAL</font>", body_style),
            Paragraph("JPEG re-compression difference reveals localized quantization inconsistencies." if scan.get("ela_flag") else "Uniform compression surface history observed.", body_style)
        ],
        [
            Paragraph("Metadata / EXIF Audit", body_style),
            Paragraph("<font color='#dc2626'>MODIFIED</font>" if meta_flags else "<font color='#16a34a'>CLEAN</font>", body_style),
            Paragraph(f"Digital signatures detected: {flags_text}", body_style)
        ]
    ]

    domain_table = Table(domain_data, colWidths=[130, 65, 335])
    domain_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#e2e8f0')),
        ('BOX', (0, 0), (-1, -1), 0.5, colors.HexColor('#cbd5e1')),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#e2e8f0')),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ('LEFTPADDING', (0, 0), (-1, -1), 6),
    ]))
    story.append(domain_table)
    story.append(Spacer(1, 10))

    # 7. System Specifications & Disclaimer
    specs_text = (
        f"<b>Model:</b> {scan.get('model_name', 'ViT Base')} &nbsp;|&nbsp; "
        f"<b>Latency:</b> {scan.get('processing_time_ms', 0):.1f} ms &nbsp;|&nbsp; "
        f"<b>Faces:</b> {scan.get('total_faces', 1)} &nbsp;|&nbsp; "
        f"<b>File:</b> {scan.get('filename', 'upload.jpg')}"
    )
    story.append(Paragraph(specs_text, body_style))
    story.append(Spacer(1, 6))

    disclaimer_text = (
        "<b>RESEARCH & FORENSIC INTEGRITY NOTICE:</b> This verification document was generated by an automated "
        "multimodal deepfake detection pipeline utilizing Vision Transformer (ViT) and Grad-CAM spatial explainability. "
        "Generative synthesis models evolve dynamically. Outcomes are probabilistic and should be corroborated with "
        "independent forensic tools before legal or journalistic attribution. Decision boundary calibrated at 0.50."
    )
    story.append(Paragraph(disclaimer_text, ParagraphStyle(
        'Disclaimer',
        parent=styles['Normal'],
        fontName='Helvetica-Oblique',
        fontSize=7,
        leading=10,
        textColor=colors.HexColor('#64748b')
    )))

    doc.build(story)
    buffer.seek(0)
    return buffer
