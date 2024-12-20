from html import escape

def generate_html_content(prompts, outputs_by_prompt, custom_ban_words):
    def is_word_allowed(output_text, banned_words):
        for word in banned_words:
            if word in output_text:
                return False
        return True

    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Generated Text</title>
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
        <style>
            body {
                font-family: Arial, sans-serif;
                background-color: #f8f9fa;
            }
            .container {
                max-width: 800px; /* Restrict max width */
                margin: auto; /* Center the container */
            }
        </style>
    </head>
    <body class="bg-light text-dark">
        <div class="container py-5">
            <h1 class="mb-4 text-center">Text Generation Output</h1>
    """
    for prompt, outputs in zip(prompts, outputs_by_prompt):
        html_content += f"""
            <div class="card mb-4 shadow-sm">
                <div class="card-body">
                    <h5 class="card-title">Prompt:</h5>
                    <p class="card-text"><strong>{escape(prompt.strip())}</strong></p>
                    <h6>Generated Texts:</h6>
                    <ul class="list-group list-group-flush">
        """
        for i, output_text in enumerate(outputs):
            if is_word_allowed(output_text, custom_ban_words):
                html_content += f"""
                    <li class="list-group-item">
                        <strong>Generated Text {i+1}:</strong> {escape(output_text)}
                    </li>
                """
            else:
                html_content += f"""
                    <li class="list-group-item text-danger">
                        <strong>Generated Text {i+1}:</strong> [Skipped due to banned words]
                    </li>
                """
        html_content += """
                    </ul>
                </div>
            </div>
        """
    html_content += """
        </div>
    </body>
    </html>
    """
    return html_content
