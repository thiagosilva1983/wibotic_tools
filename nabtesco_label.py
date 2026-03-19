import re
import tkinter as tk
from tkinter import filedialog, messagebox, font
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import inch
import os
import csv
import tempfile
from collections import Counter
from PIL import Image

class LabelApp:
    def __init__(self, root):
        self.root = root
        self.root.title("4x2.5 Label Creator")
        self.labels = []
        self.printed_labels = set()
        self.processed_serials = set()
        self.font_size = tk.IntVar(value=10)
        self.label_width = tk.DoubleVar(value=4.0)
        self.label_height = tk.DoubleVar(value=2.32)
        self.default_save_location = os.path.expanduser("~/Desktop/labels.pdf")
        self.sales_order = ""
        self.customer_purchase_order = ""

        # Logo file in same folder as this script
        self.logo_path = os.path.join(os.path.dirname(__file__), "Wibotic 2.png")

        self.default_fields = {"Part Number": "", "Part Name": "", "Serial Number": "", "Quantity": "1"}
        self.item_summary = Counter()
        self.create_gui()

    def create_gui(self):
        mono_font = font.Font(family="Courier", size=10)
        self.label_listbox = tk.Listbox(self.root, width=120, height=15, font=mono_font)
        self.label_listbox.pack(padx=10, pady=10)

        sales_order_frame = tk.Frame(self.root)
        sales_order_frame.pack(pady=5)
        tk.Label(sales_order_frame, text="Sales Order:").grid(row=0, column=0, padx=5)
        self.sales_order_entry = tk.Entry(sales_order_frame, width=30)
        self.sales_order_entry.grid(row=0, column=1, padx=5)

        customer_po_frame = tk.Frame(self.root)
        customer_po_frame.pack(pady=5)
        tk.Label(customer_po_frame, text="Customer Purchase Order:").grid(row=0, column=0, padx=5)
        self.customer_po_entry = tk.Entry(customer_po_frame, width=30)
        self.customer_po_entry.grid(row=0, column=1, padx=5)

        button_frame = tk.Frame(self.root)
        button_frame.pack()
        tk.Button(button_frame, text="Upload CSV", command=self.upload_csv).grid(row=0, column=0, padx=5)
        tk.Button(button_frame, text="Generate PDF", command=self.generate_pdf).grid(row=0, column=1, padx=5)
        tk.Button(button_frame, text="Print Selected Label", command=self.print_selected_label).grid(row=0, column=2, padx=5)
        tk.Button(button_frame, text="Help / How to Use", command=self.show_help).grid(row=0, column=3, padx=5)

    def show_help(self):
        instructions = (
            "📦 How to Generate the CSV for Label Printing:\n\n"
            "1. Open SOS and go to the Sales Order.\n"
            "2. Click on the shipment.\n"
            "3. Click the down arrow, then select 'Quick View'.\n"
            "4. Copy the table and paste it into a spreadsheet.\n"
            "5. DELETE the 'Backorder' column.\n"
            "6. Save or download the spreadsheet as CSV.\n"
            "7. Use the 'Upload CSV' button in this app.\n\n"
            "✅ Column names accepted (aliases work):\n"
            "   - Item / Part Number\n"
            "   - Description / Part Name\n"
            "   - Shipped / Quantity\n"
            "   - Serial Number / S/N / SN (optional if SN is in Description)\n"
        )
        messagebox.showinfo("Help / How to Use", instructions)

    # Helper: clean “Part Name” by removing SN/S/N/MAC tokens
    def _clean_part_name(self, text: str) -> str:
        # remove blocks like "S/N: ....", "SN: ....", "MAC: ...." up to '|' or EOL
        t = re.sub(r'\b(?:S\/?N|SN|MAC)\b\s*[:\-]?\s*[^|\r\n]+', '', text, flags=re.IGNORECASE)
        # collapse leftover spaces and separators
        t = re.sub(r'\s*\|\s*', ' ', t)
        t = re.sub(r'\s{2,}', ' ', t).strip(' -|')
        return t.strip()

    def upload_csv(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
        if not file_path:
            return
        try:
            self.labels.clear()
            self.printed_labels.clear()
            self.processed_serials.clear()
            self.item_summary.clear()
            added_unserialized_parts = set()

            with open(file_path, "r", newline="", encoding="utf-8-sig") as csv_file:
                reader = csv.DictReader(csv_file)
                for row in reader:
                    part_number = (row.get("Item") or row.get("Part Number") or "Unknown").strip()
                    desc_raw    = (row.get("Description") or row.get("Part Name") or "Unknown").strip()
                    quantity_str = (row.get("Shipped") or row.get("Quantity") or "1").strip()

                    # Build a nice Part Name:
                    # - prefer MN: value if present; else Description with SN/MAC stripped.
                    m_mn = re.search(r'\bMN\b\s*[:\-]?\s*([^|\s,;]+)', desc_raw, re.IGNORECASE)
                    part_name = m_mn.group(1).strip() if m_mn else self._clean_part_name(desc_raw)

                    serialized = False

                    # 1) Extract all serials from Description
                    sn_blocks = re.findall(r'(?:\bS\/?N\b|\bSN\b)\s*[:\-]?\s*([^|\r\n]+)', desc_raw, flags=re.IGNORECASE)
                    for block in sn_blocks:
                        for serial_number in re.split(r'[,\s;|]+', block.strip()):
                            serial_number = serial_number.strip()
                            if not serial_number:
                                continue
                            if serial_number not in self.processed_serials:
                                self.processed_serials.add(serial_number)
                                fields = self.default_fields.copy()
                                fields["Part Number"] = part_number
                                fields["Part Name"] = part_name
                                fields["Serial Number"] = serial_number
                                fields["Quantity"] = "1"
                                self.labels.append(fields)
                                self.item_summary[(fields["Part Number"], fields["Part Name"])] += 1
                                serialized = True

                    # 2) Fallback: explicit SN column
                    if not serialized:
                        raw_sn = (row.get("Serial Number") or row.get("S/N") or row.get("SN") or "").strip()
                        if raw_sn:
                            for serial_number in re.split(r'[,\s;|]+', raw_sn):
                                serial_number = serial_number.strip()
                                if serial_number and serial_number not in self.processed_serials:
                                    self.processed_serials.add(serial_number)
                                    fields = self.default_fields.copy()
                                    fields["Part Number"] = part_number
                                    fields["Part Name"] = part_name
                                    fields["Serial Number"] = serial_number
                                    fields["Quantity"] = "1"
                                    self.labels.append(fields)
                                    self.item_summary[(fields["Part Number"], fields["Part Name"])] += 1
                                    serialized = True

                    # 3) Non-serialized: one label with qty
                    if not serialized and part_number not in added_unserialized_parts:
                        fields = self.default_fields.copy()
                        fields["Part Number"] = part_number
                        fields["Part Name"] = part_name
                        fields["Serial Number"] = "N/A"
                        fields["Quantity"] = quantity_str
                        self.labels.append(fields)
                        self.item_summary[(fields["Part Number"], fields["Part Name"])] += 1
                        added_unserialized_parts.add(part_number)

            self.refresh_listbox()
            messagebox.showinfo("Success", "Labels successfully loaded from CSV.")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to upload CSV: {e}")

    def generate_pdf(self):
        if not self.labels:
            messagebox.showwarning("No Labels", "No labels to generate PDF.")
            return

        self.sales_order = self.sales_order_entry.get().strip()
        self.customer_purchase_order = self.customer_po_entry.get().strip()

        if not self.sales_order or not self.customer_purchase_order:
            messagebox.showwarning("Missing Info", "Sales Order and Customer PO are required.")
            return

        file_path = filedialog.asksaveasfilename(
            initialfile=self.default_save_location,
            defaultextension=".pdf",
            filetypes=[("PDF Files", "*.pdf")]
        )
        if not file_path:
            return

        font_size = self.font_size.get()
        label_width = self.label_width.get() * inch
        label_height = self.label_height.get() * inch

        try:
            c = canvas.Canvas(file_path, pagesize=(label_width, label_height))
            for label in self.labels:
                self.draw_label_on_canvas(c, label, label_width, label_height, font_size)
                c.showPage()
            c.save()
            messagebox.showinfo("Success", f"PDF saved at: {os.path.abspath(file_path)}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to create PDF: {e}")

    def print_selected_label(self):
        selected_index = self.label_listbox.curselection()
        if not selected_index:
            messagebox.showwarning("No Selection", "Please select a label to print.")
            return

        label = self.labels[selected_index[0]]
        self.sales_order = self.sales_order_entry.get().strip()
        self.customer_purchase_order = self.customer_po_entry.get().strip()

        if not self.sales_order or not self.customer_purchase_order:
            messagebox.showwarning("Missing Info", "Sales Order and Customer PO are required.")
            return

        font_size = self.font_size.get()
        label_width = self.label_width.get() * inch
        label_height = self.label_height.get() * inch

        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                c = canvas.Canvas(tmp_file.name, pagesize=(label_width, label_height))
                self.draw_label_on_canvas(c, label, label_width, label_height, font_size)
                c.save()
                os.startfile(tmp_file.name)  # Windows only

                key = f"{label['Part Number']}-{label['Serial Number']}"
                self.printed_labels.add(key)
                self.refresh_listbox()
        except Exception as e:
            messagebox.showerror("Print Error", f"Failed to print label: {e}")

    def draw_label_on_canvas(self, c, label, label_width, label_height, font_size):
        c.setFont("Helvetica-Bold", font_size)
        y_position = label_height - 0.3 * inch

        if self.logo_path and os.path.exists(self.logo_path):
            try:
                with Image.open(self.logo_path) as img:
                    img_width, img_height = img.size
                    max_width = 0.6 * inch
                    max_height = 0.6 * inch
                    ratio = min(max_width / img_width, max_height / img_height)
                    new_width = img_width * ratio
                    new_height = img_height * ratio
                    c.drawImage(
                        self.logo_path,
                        label_width - new_width - 0.2 * inch,
                        label_height - new_height - 0.2 * inch,
                        width=new_width,
                        height=new_height,
                        preserveAspectRatio=True
                    )
            except:
                pass

        c.drawString(0.5 * inch, y_position, f"Sales Order: {self.sales_order}")
        y_position -= 0.3 * inch
        c.drawString(0.5 * inch, y_position, f"Customer Purchase Order: {self.customer_purchase_order}")
        y_position -= 0.3 * inch

        # Print fields in fixed order so layout is predictable
        for key in ["Part Number", "Part Name", "Serial Number", "Quantity"]:
            value = label.get(key, "")
            c.drawString(0.5 * inch, y_position, f"{key}: {value}")
            y_position -= 0.3 * inch

    def refresh_listbox(self):
        self.label_listbox.delete(0, tk.END)
        for label in self.labels:
            key = f"{label['Part Number']}-{label['Serial Number']}"
            printed_marker = "✓" if key in self.printed_labels else ""
            label_text = (
                f"PN: {label['Part Number']:<15} | "
                f"Name: {label['Part Name'][:25]:<25} | "
                f"SN: {label['Serial Number']:<15} | "
                f"Qty: {label['Quantity']:<3} {printed_marker}"
            )
            self.label_listbox.insert(tk.END, label_text)

if __name__ == "__main__":
    root = tk.Tk()
    app = LabelApp(root)
    root.mainloop()
