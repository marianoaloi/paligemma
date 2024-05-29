

from flask import Flask, render_template, request, send_file


class WebPaligemma:
    app = Flask(__name__, 
            static_folder='web/static',
            template_folder='web/templates')

    def __init__(self) -> None:
        pass

        @self.app.route('/')
        def hello_word():
            return render_template('screen.html')

        @self.app.route('/img/')
        def get_image():
            filename = request.args.get("path")  # Replace with your image filename
            return send_file(filename, mimetype='image/jpeg')

if __name__ == "__main__":
    web=WebPaligemma()
    web.app.run(debug=False,host="0.0.0.0")