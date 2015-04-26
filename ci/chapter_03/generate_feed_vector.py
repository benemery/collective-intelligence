from collections import defaultdict
import re

import feedparser

def get_word_counts(url):
    """Return a tuple of feed title and word counts."""
    data = feedparser.parse(url)
    word_counts = defaultdict(int)

    for entry in data.entries:
        summary = entry.get('summary', entry.description)

        words = get_words(entry.title + ' ' + summary)
        for word in words:
            word_counts[word] += 1

    title = data.feed.get('title', '')
    if not title:
        title = url
    return title, word_counts


def get_words(html):
    """Parse HTML and return a list of words"""
    # Strip tags
    text = re.sub(r'<[^>]+>', '', html)
    # Split on anything that's non-alpha
    words = re.split(r'[^A-Z^a-z]+', text)

    return [word.lower() for word in words if word]


def generate_vector(stream, lower_bound=0.1, upper_bound=0.5, verbose=True):
    """Given a stream of URLs, build a word"""
    appear_count = defaultdict(int)
    all_word_counts = {}
    feed_list = []

    # Count all the words in all the rss feeds
    for url in stream:
        url = url.strip()
        if verbose:
            # Print the URL we're working so you know it's not hanging
            print url

        feed_list.append(url)
        title, word_counts = get_word_counts(url)
        all_word_counts[title] = word_counts

        for word, count in word_counts.items():
            if count > 1:
                appear_count[word] += 1

    # Strip out common / rare words
    word_list = []
    for word, count in appear_count.items():
        appear_precentage = 1.0 * count / len(feed_list)
        if lower_bound < appear_precentage < upper_bound:
            word_list.append(word)

    with open('blogdata.txt', 'wb') as out:
        out.write('Blog')
        for word in word_list:
            out.write('\t%s' % word)
        out.write('\n')

        for text, word_count in all_word_counts.items():
            text = text.encode('ascii', 'ignore')
            out.write(text)
            for word in word_list:
                if word in word_count:
                    out.write('\t%d' % word_count[word])
                else:
                    out.write('\t0')
            out.write('\n')


if __name__ == '__main__':
    with open('feedlist.txt', 'rb') as fin:
        generate_vector(fin)
