import gpt_2_simple as gpt2
from flask import Flask, render_template, url_for, request

app = Flask(__name__)


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/predict', methods=['POST'])
def predict():
    sess = gpt2.start_tf_sess()
    gpt2.load_gpt2(sess)

    if request.method == 'POST':
        prefix = request.form['message']
        text = gpt2.generate(sess,
                             length=40,
                             temperature=0.7,
                             prefix=prefix,
                             nsamples=1,
                             batch_size=1,
                             return_as_list=True
                             )

        t = text[0].title()
        t = t.replace('<|Startoftext|>', '').replace(
            '\n', '')  # remove extraneous stuff
        t = t[:t.index('<|Endoftext|>')]  # only get one title
        return render_template('result.html', prediction=t)


if __name__ == '__main__':
    app.run(debug=True)
