# coding: utf-8

# ## Practical 3: Text Classification with RNNs
# <p>Oxford CS - Deep NLP 2017<br>
# https://www.cs.ox.ac.uk/teaching/courses/2016-2017/dl/</p>
# <p>[Chris Dyer, Phil Blunsom, Yannis Assael, Brendan Shillingford, Yishu Miao]</p>

import numpy as np
import os
from random import shuffle
import re
import collections
import tensorflow as tf
import urllib.request
import zipfile
import lxml.etree


# Download the dataset if it's not already there: this may take a minute as it is 75MB
if not os.path.isfile('ted_en-20160408.zip'):
    urllib.request.urlretrieve("https://wit3.fbk.eu/get.php?path=XML_releases/xml/ted_en-20160408.zip&filename=ted_en-20160408.zip", filename="ted_en-20160408.zip")


with zipfile.ZipFile('ted_en-20160408.zip', 'r') as z:
    doc = lxml.etree.parse(z.open('ted_en-20160408.xml', 'r'))

print("data downloaded")

root = doc.getroot()

del doc

labels_talks_ted = []
tokens_talks_ted = []

for file in root:
    keywords = file.find("head").find("keywords").text.lower()
    content = file.find("content").text.lower()
    label = np.empty(shape=(1,3), dtype="float16")
    label[0,0]= (1 if keywords.find("technology")>-1 else 0)
    label[0,1]= (1 if keywords.find("entertainment")>-1 else 0)
    label[0,2]= (1 if keywords.find("design")>-1 else 0)
    
    content = re.sub(r'\([^)]*\)', '', content)
    
    sentences_strings_content = []
    for line in content.split('\n'):
        m = re.match(r'^(?:(?P<precolon>[^:]{,20}):)?(?P<postcolon>.*)$', line)
        sentences_strings_content.extend(sent for sent in m.groupdict()['postcolon'].split('.') if sent)
    
    content_tokens = []
    for sent_str in sentences_strings_content:
        tokens = re.sub(r"[^a-z0-9]+", " ", sent_str.lower()).split()
        content_tokens += tokens
    
    labels_talks_ted.append(label)
    tokens_talks_ted.append(content_tokens)

min_count = 10

training_labels_talks_ted = labels_talks_ted[0:1585]
training_tokens_talks_ted = tokens_talks_ted[0:1585]

training_counts_ted = collections.Counter()
training_tokens_ted = [token for talk in training_tokens_talks_ted for token in talk]

for token in training_tokens_ted:
    training_counts_ted[token] += 1

training_idx_ted = {}

next_idx = 0
for talk in training_tokens_talks_ted:
    for idx in range(0, len(talk)):
        if training_counts_ted[talk[idx]] < min_count:
            talk[idx]="UNKNOWNTEXT"
        if talk[idx] in training_idx_ted:
            continue
        else:
            training_idx_ted[talk[idx]] = next_idx
            next_idx+=1
            
vocab_size = len(training_idx_ted)

print("data preprocessed")

def length(sequence):
    used = tf.sign(tf.reduce_max(tf.abs(sequence), reduction_indices=2))
    length = tf.reduce_sum(used, reduction_indices=1)
    length = tf.cast(length, tf.int32)
    return length

init_scale = 0.1
learning_rate = .001
max_grad_norm = 5
num_layers = 3
num_steps = 1000
num_outputs = 3
hidden_size = 500
keep_prob = 1.0
batch_size = 50

x = tf.placeholder("float16",shape=[None, num_steps, vocab_size], name="x_placeholder")
y = tf.placeholder("float16",shape=[None, num_outputs], name="y_placeholder")
weights = tf.Variable(tf.truncated_normal([hidden_size, num_outputs], stddev=0.05, dtype=tf.float16))
bias = tf.Variable(tf.constant(.1, shape=[num_outputs], dtype=tf.float16))

lstm = tf.contrib.rnn.BasicLSTMCell(hidden_size, forget_bias=0.0, state_is_tuple=True)
stacked_lstm = tf.contrib.rnn.MultiRNNCell([lstm] * num_layers, state_is_tuple=True)

initial_state = stacked_lstm.zero_state(batch_size, dtype=tf.float16)

outputs, state = tf.nn.dynamic_rnn(stacked_lstm, x, initial_state=initial_state, dtype=tf.float16, sequence_length=length(x))

# outputs = tf.transpose(outputs, [1,0,2])
# last = tf.gather(outputs, num_steps - 1)
# y_pred = tf.nn.softmax(tf.matmul(last, weights) + bias)
outputs = tf.reduce_mean(outputs, 1)
y_pred = tf.nn.softmax(tf.matmul(outputs, weights) + bias)

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_pred, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cross_entropy)

correct_pred = tf.equal(tf.argmax(y_pred,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

print("graph created")

# format all data
sequences = np.empty(shape=(len(training_tokens_talks_ted),num_steps,vocab_size), dtype="float16")
            
for talk_idx in range(0, len(training_tokens_talks_ted)):
    talk = training_tokens_talks_ted[talk_idx]
    for token_idx in range(0,min(num_steps,len(talk))):
        token = training_tokens_talks_ted[talk_idx][token_idx]
        sequences[talk_idx][token_idx][training_idx_ted[token]] = 1
		
    print("sequence generated for talk %d/%d"%(talk_idx, len(training_tokens_talks_ted)))

labels = np.reshape(training_labels_talks_ted, (len(training_labels_talks_ted),num_outputs))

max_epochs = 10
epoch_iterations = len(sequences) / batch_size
num_iterations = int(max_epochs * epoch_iterations)


def data_iterator():
    batch_idx = 0
    while True:
        for batch_idx in range(0, len(training_tokens_talks_ted), batch_size):
            sequence_batch = sequences[batch_idx:batch_idx+batch_size]
            labels_batch = labels[batch_idx:batch_idx+batch_size]
            yield sequence_batch, labels_batch
            
iter_ = data_iterator()

saver = tf.train.Saver()

init = tf.global_variables_initializer()

print("training...")

with tf.Session() as sess:
    sess.run(init)
    step = 1
    for i in range(num_iterations):
        sequence_batch, labels_batch = iter_.__next__()
        if (i+1)%10==0:
            train_accuracy = accuracy.eval(session = sess, feed_dict={ x:sequence_batch, y: labels_batch})
            print("step %d, training accuracy %g"%(i, train_accuracy))
        optimizer.run(session = sess, feed_dict={x: sequence_batch, y: labels_batch})
        if (i+1)%epoch_iterations==0:
            save_path = saver.save(sess, "/tmp/model.ckpt")
            print("Model saved in file: %s"%save_path)

# create the sparse matrix first then fill it in... good idea...
