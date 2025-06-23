from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QTextEdit


class ChatWidget(QWidget):
    """Виджет для отображения чата"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout(self)
        self.chat_text = QTextEdit()
        self.chat_text.setReadOnly(True)
        self.chat_text.setFont(QFont("Consolas", 10))
        self.chat_text.setMinimumHeight(400)
        self.chat_text.setStyleSheet("""
            QTextEdit {
                background-color: #3C3C3C;
                color: #E0E0E0;
                border: none;
                border-radius: 5px;
                padding: 10px;
                line-height: 1.4;
            }
            QTextEdit QScrollBar:vertical {
                background-color: #2C2C2C;
                width: 12px;
                border-radius: 6px;
            }
            QTextEdit QScrollBar::handle:vertical {
                background-color: #6C6C6C;
                border-radius: 6px;
                min-height: 20px;
            }
            QTextEdit QScrollBar::handle:vertical:hover {
                background-color: #8C8C8C;
            }
        """)
        layout.addWidget(self.chat_text)

    def add_message(self, user_text: str, ai_response: str):
        cursor = self.chat_text.textCursor()
        cursor.movePosition(cursor.MoveOperation.End)
        cursor.insertHtml(f'<p style="color: #E0E0E0; margin: 5px 0;"><b>👤 User:</b> {user_text}</p>')
        formatted_response = self._format_code_blocks(ai_response)
        cursor.insertHtml(f'<p style="color: #E0E0E0; margin: 0 0 15px 0;"><br><b>🤖 AI:</b> {formatted_response}</p>')
        self.chat_text.setTextCursor(cursor)
        self.chat_text.ensureCursorVisible()

    def _format_code_blocks(self, text: str) -> str:
        import re
        text = text.replace('\n', '<br>')
        code_block_pattern = r'```(\w+)?\s*\n(.*?)\n```'

        def replace_code_block(match):
            language = match.group(1) or 'text'
            code_content = match.group(2)
            highlighted_code = self._highlight_syntax(code_content, language)
            formatted_code = f'<div style="background-color: #1E1E1E; border: 1px solid #6C6C6C; border-radius: 5px; padding: 10px; margin: 10px 0; font-family: Consolas, monospace; font-size: 11px; color: #D4D4D4; white-space: pre-wrap;">'
            formatted_code += f'<div style="color: #569CD6; font-weight: bold; margin-bottom: 5px;">{language.upper()}</div>'
            formatted_code += f'<div style="color: #D4D4D4;">{highlighted_code}</div>'
            formatted_code += '</div>'
            return formatted_code

        text = re.sub(code_block_pattern, replace_code_block, text, flags=re.DOTALL)
        inline_code_pattern = r'`([^`]+)`'

        def replace_inline_code(match):
            code_content = match.group(1)
            return f'<span style="background-color: #3C3C3C; color: #D4D4D4; font-family: Consolas, monospace; padding: 2px 4px; border-radius: 3px; font-size: 11px;">{code_content}</span>'

        text = re.sub(inline_code_pattern, replace_inline_code, text)
        return text

    def _highlight_syntax(self, code: str, language: str) -> str:
        import re
        colors = {
            'keyword': '#569CD6',
            'string': '#CE9178',
            'comment': '#6A9955',
            'number': '#B5CEA8',
            'function': '#DCDCAA',
            'default': '#D4D4D4'
        }
        patterns = {
            'python': {
                'keyword': r'\b(def|class|import|from|as|if|else|elif|for|while|try|except|finally|with|return|True|False|None|and|or|not|in|is|lambda|yield|async|await)\b',
                'string': r'"[^"]*"|\'[^"]*\'',
                'comment': r'#.*$',
                'function': r'\b\w+(?=\()',
                'number': r'\b\d+\.?\d*\b',
            },
            'javascript': {
                'keyword': r'\b(function|var|let|const|if|else|for|while|try|catch|finally|return|class|extends|import|export|async|await|new|this|super)\b',
                'string': r'"[^"]*"|\'[^"]*\'|`[^`]*`',
                'comment': r'//.*$|/\*.*?\*/',
                'function': r'\b\w+(?=\()',
                'number': r'\b\d+\.?\d*\b',
            },
            'java': {
                'keyword': r'\b(public|private|protected|static|final|class|interface|extends|implements|if|else|for|while|try|catch|finally|return|new|this|super|import|package)\b',
                'string': r'"[^"]*"',
                'comment': r'//.*$|/\*.*?\*/',
                'function': r'\b\w+(?=\()',
                'number': r'\b\d+\.?\d*\b',
            },
            'cpp': {
                'keyword': r'\b(int|float|double|char|bool|void|class|struct|enum|public|private|protected|static|const|virtual|template|namespace|using|include|#include|#define|#ifdef|#endif)\b',
                'string': r'"[^"]*"',
                'comment': r'//.*$|/\*.*?\*/',
                'function': r'\b\w+(?=\()',
                'number': r'\b\d+\.?\d*\b',
            }
        }
        if language.lower() not in patterns:
            return code
        lang_patterns = patterns[language.lower()]
        highlighted_code = code
        for token_type, pattern in lang_patterns.items():
            color = colors.get(token_type, colors['default'])

            def replace_token(match):
                return f'<span style="color: {color};">{match.group(0)}</span>'

            highlighted_code = re.sub(pattern, replace_token, highlighted_code, flags=re.MULTILINE)
        return highlighted_code

    def clear(self):
        self.chat_text.clear()
