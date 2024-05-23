from flask import Flask

class WebPaligemma:
    app = Flask(__name__)

    def __init__(self) -> None:
        pass

        @self.app.route('/')
        def hello_word():
            return "<p>Hello, world!</p>"



if __name__ == "__main__":
    web=WebPaligemma()
    web.app.run(debug=True)