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

print("content tokenized and cleaned")

# Count each token in training talks

training_counts_ted = collections.Counter()
training_tokens_ted = [token for talk in tokens_talks_ted[0:1585] for token in talk]

for token in training_tokens_ted:
    training_counts_ted[token] += 1

print("tokens counted")

# Determine onehot encoding idx for each token
# Remove all unknown tokens

onehot_idx_ted = {}

min_count = 10
next_idx = 0
for talk in tokens_talks_ted:
    for idx in range(0, len(talk)):
        if talk[idx] not in training_counts_ted or training_counts_ted[talk[idx]] < min_count:
            talk[idx]="UNKNOWNTEXT"
        if talk[idx] in onehot_idx_ted:
            continue
        else:
            onehot_idx_ted[talk[idx]] = next_idx
            next_idx+=1
            
vocab_size = len(onehot_idx_ted)

print("unknown tokens removed")

# Create One-Hot Encodings for all tokens

batch_size = 50
num_outputs = 3
num_steps = 1000

sequences = np.empty(shape=(len(tokens_talks_ted),num_steps,vocab_size), dtype="float16")
            
for talk_idx in range(0, len(tokens_talks_ted)):
    talk = tokens_talks_ted[talk_idx]
    for token_idx in range(0,min(num_steps,len(talk))):
        token = tokens_talks_ted[talk_idx][token_idx]
        sequences[talk_idx][token_idx][onehot_idx_ted[token]] = 1
		
labels = np.reshape(labels_talks_ted, (len(labels_talks_ted),num_outputs))

print ("one hot encodings created")

training_sequences = sequences[0:1585]
training_labels = labels[0:1585]

validation_sequences = sequences[1585:1835]
validation_labels = labels[1585:1835]

testing_sequences = sequences[1835:2085]
testing_labels = labels[1835:2085]

# Build LSTM graph

init_scale = 0.1
learning_rate = .001
max_grad_norm = 5
num_layers = 3
hidden_size = 500
keep_prob = 1.0

def length(sequence):
    used = tf.sign(tf.reduce_max(tf.abs(sequence), reduction_indices=2))
    length = tf.reduce_sum(used, reduction_indices=1)
    length = tf.cast(length, tf.int32)
    return length

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

max_epochs = 10
training_epoch_iterations = int(len(training_sequences) / batch_size)
validation_epoch_iterations = int(len(validation_sequences) / batch_size)
testing_epoch_iterations = int(len(testing_sequences) / batch_size)
training_iterations = int(max_epochs * training_epoch_iterations)

# Create iterator

def data_iterator(sequences, labels, batch_size):
    batch_idx = 0
    while True:
        for batch_idx in range(0, len(sequences), batch_size):
            if batch_idx+batch_size+1 > len(sequences):
                continue
            sequences_batch = sequences[batch_idx:batch_idx+batch_size]
            labels_batch = labels[batch_idx:batch_idx+batch_size]
            yield sequences_batch, labels_batch
            
training_iter_ = data_iterator(training_sequences, training_labels, batch_size)

validation_iter_ = data_iterator(validation_sequences, validation_labels, batch_size)

testing_iter_ = data_iterator(testing_sequences, testing_labels, batch_size)

saver = tf.train.Saver()

init = tf.global_variables_initializer()

print("training...")

with tf.Session() as sess:
    sess.run(init)
    step = 1
    for i in range(training_iterations):
        training_sequences_batch, training_labels_batch = training_iter_.__next__()
        # periodically print training error

        if (i+1)%10==0:
            train_accuracy = accuracy.eval(session = sess, feed_dict={ x:training_sequences_batch, y: training_labels_batch})
            print("step %d, training accuracy %g"%(i+1, train_accuracy))

        optimizer.run(session = sess, feed_dict={x: training_sequences_batch, y: training_labels_batch})

        # periodically save model and print validation error

        if (i+1)%training_epoch_iterations==0:
            for j in range(validation_epoch_iterations):
                validation_sequences_batch, validation_labels_batch = validation_iter_.__next__()
                validation_accuracy = accuracy.eval(session=sess, feed_dict={ x: validation_sequences_batch, y: validation_labels_batch})
            print("epoch %d, validation accuracy %g"%((i+1)/training_epoch_iterations, validation_accuracy))

            save_path = saver.save(sess, "tmp/model_%d.ckpt"%(i+1))
            print("Model saved in file: %s"%save_path)

    for i in range(testing_epoch_iterations):
        testing_sequences_batch, testing_labels_batch = testing_iter_.__next__()
        testing_accuracy = accuracy.eval(session=sess, feed_dict={ x: testing_sequences_batch, y: testing_labels_batch})
        print("testing accuracy %g"%(testing_accuracy))
