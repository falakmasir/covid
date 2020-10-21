import pandas as pd
import numpy as np
import warnings
import json
import datetime
import spacy
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import plotly.graph_objects as go
from palettable.scientific import sequential
from pynytimes import NYTAPI
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation
from sklearn.preprocessing import normalize
pd.options.mode.chained_assignment = None
warnings.filterwarnings('ignore')

external_stylesheets = ['https://fonts.googleapis.com/css?family=Open+Sans:300,400,700',
                        'https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

app.css.config.serve_locally = True
app.scripts.config.serve_locally = True
app.config.suppress_callback_exceptions = True

server = app.server

app.title = 'Topic Modeling'

nlp = spacy.load('en_core_web_lg')

app.layout = html.Div(children=[

    ####################################################################################################################
    # header
    ####################################################################################################################

    html.Div(children=[

        html.H5(children=['COVID-19 Publication Dashboard'], style={'display': 'inline-block', 'margin-left': '2vw'}),

    ], className='row', style={'background-color': '#421E4A', 'color': 'white', 'width': '100%'}),

    ####################################################################################################################
    # first column
    ####################################################################################################################

    html.Div(children=[

        # data
        html.Label('Data', style={'margin': '1vw 0vw 0.25vw 0vw', 'color': '#673762'}),

        # source
        html.Label(children=['Source: MIDAS Coordination Center'], style={'font-size': '90%'}),

        # number of articles
        html.Div(children=['Number of articles: ', html.Span(id='count')],
        style={'font-size': '90%', 'white-space': 'pre'}),

        # start date
        html.Div(children=['Start Date: ', html.Span(id='start_date')],
        style={'font-size': '90%', 'white-space': 'pre'}),

        # end date
        html.Div(children=['End Date: ', html.Span(id='end_date')],
        style={'font-size': '90%', 'white-space': 'pre'}),

        # model
        html.Label(children=['Model'], style={'margin': '1vw 0vw 0.25vw 0vw', 'color': '#673762'}),

        # radio items
        dcc.RadioItems(id='model',
                       options=[{'label': 'Nonnegative Matrix Factorization', 'value': 'nmf'},
                                {'label': 'Probabilistic Latent Semantic Analysis', 'value': 'plsa'},
                                {'label': 'Latent Dirichlet Allocation', 'value': 'lda'}],
                       value='nmf',
                       style={'font-size': '90%'}),

        # components
        html.Label(children=['Components'], style={'margin': '1vw 0vw 0.25vw 0vw', 'color': '#673762'}),

        # numeric input
        dcc.Input(id='components',
                  type='number',
                  value=3,
                  min=2,
                  max=5,
                  style={'width': '8vw', 'margin': '0vw 0vw 1vw 0vw'}),

        # update button
        html.Button(id='update', children=['Update'], style={'width': '10vw', 'height': '2.25vw',
        'line-height': '2.25vw', 'font-size': '0.6vw', 'display': 'block', 'margin': '1.5vw 0vw 1.5vw 0vw',
        'background-color': '#8B6F7F', 'color': 'white', 'border-color': '#d9d9d9'}),

        # about
        # html.Label(children=['About'], style={'margin': '1vw 0vw 0.25vw 0vw', 'color': '#673762'}),

        # # reference
        # html.P(children=['The design of this app is broadly based on Y. Yang, Q. Yao, H. Qu (2017), \"VISTopic: '
        # 'A visual analytics system for making sense of large document collections using hierarchical topic '
        # 'modeling\",', html.I(' Visual Informatics'), ', Volume 1, p. 40-47.'],
        # style={'font-size': '90%', 'text-align': 'justify'}),

    ], style={'display': 'inline-block', 'vertical-align': 'top', 'width': '18vw', 'margin': '0vw 0vw 1vw 2vw'}),

    ####################################################################################################################
    # second column
    ####################################################################################################################

    html.Div(children=[
        
       # first row
        html.Div(children=[

            # document view
            html.Label(children=['Document View'], style={'margin': '1vw 0vw 0.5vw 0vw', 'color': '#673762'}),

                html.Div(children=[

                    html.Div(children=[

                        # article
                        html.Div(children=[

                            # headline
                            html.Div(id='first_line', style={'font-size': '85%', 'text-align': 'justify',
                            'margin': '1vw 1vw 1vw 2vw'}),

                            # snippet
                            html.Div(id='second_line', style={'font-size': '85%', 'text-align': 'justify',
                            'margin': '0vw 1vw 1vw 2vw'}),

                            # paragraph
                            html.Div(id='third_line', style={'font-size': '85%', 'text-align': 'justify',
                            'margin': '0vw 1vw 0vw 2vw'}),

                        ], style={'display': 'inline-block', 'vertical-align': 'top', 'height': '15vw',
                        'width': '40vw', 'text-align': 'justify'}),

                        # radar chart
                        html.Div(children=[

                            dcc.Graph(id='document_view', config={'responsive': True, 'autosizable': True},
                            style={'height': '15vw', 'width': '15vw', 'margin': '0vw 0vw 0vw 5vw'}),

                        ], style={'display': 'inline-block', 'vertical-align': 'top', 'height': '15vw',
                        'width': '15vw'}),

                    ], className='row'),

                    html.Div(children=[

                        # backward button
                        html.Button(id='previous', children=['Previous'], style={'width': '10vw', 'height': '2.25vw',
                        'line-height': '2.25vw', 'font-size': '0.6vw', 'display': 'inline', 'margin': '0.5vw 0vw 1vw 2vw',
                        'background-color': '#8B6F7F', 'color': 'white', 'border-color': '#d9d9d9'}),

                        # forward button
                        html.Button(id='next', children=['Next'], style={'width': '10vw', 'height': '2.25vw',
                        'line-height': '2.25vw', 'font-size': '0.6vw', 'display': 'inline', 'margin': '0.5vw 0vw 1vw 45vw',
                        'background-color': '#8B6F7F', 'color': 'white', 'border-color': '#d9d9d9'}),

                    ], className='row'),

                ], style={'border-radius': '1rem', 'background-color': 'white'}),

            ], className='row'),

        # second row
        html.Div(children=[
            
            html.Div(children=[ 
    
                # topic view
                html.Label(children=['Topic View'], style={'margin': '1vw 0vw 0.25vw 0vw', 'color': '#673762'}),

                # sunburst chart
                dcc.Graph(id='topic_view', config={'responsive': True, 'autosizable': True},
                style={'height': '20vw', 'width': '20vw', 'margin': '1vw 0vw 0vw 2vw'}),
                            
            ], style={'display': 'inline-block', 'vertical-align': 'top', 'height': '22vw', 'width': '25vw',
            'margin': '1vw 0vw 0vw 0vw'}),

            html.Div(children=[

                # evolution view
                html.Label(children=['Evolution View'], style={'margin': '1vw 0vw 0.25vw 0vw', 'color': '#673762'}),

                # line chart
                dcc.Graph(id='evolution_view', config={'responsive': True, 'autosizable': True},
                style={'height': '20vw', 'width': '40vw', 'margin': '0.5vw 0vw 0vw 0vw'}),

            ], style={'display': 'inline-block', 'vertical-align': 'top', 'height': '22vw', 'width': '40vw',
            'margin': '1vw 0vw 0vw 4vw'}),
        
        ], className='row'),

    ], style={'display': 'inline-block', 'vertical-align': 'top', 'width': '69vw', 'margin': '0vw 0vw 0vw 6vw'}),

    # hidden divs
    html.Div(id='articles_counter', style={'display': 'none'}),
    html.Div(id='articles_data', style={'display': 'none'}),

])

@app.callback([Output('count', 'children'), Output('start_date', 'children'), Output('end_date', 'children'),
    Output('topic_view', 'figure'), Output('evolution_view', 'figure'), Output('articles_data', 'children')],
    [Input('update', 'n_clicks')], [State('model', 'value'), State('components', 'value')])

def update_graphs(n_clicks, selected_model, n_components):

    df = pd.read_pickle('./covid_dash.pkl')
    df['date'] = pd.to_datetime(df['date'])
    df.sort_values(by='date', inplace=True)

    print('----reading the file ----')


    count = format(df.shape[0], ',d') # number of articles
    start_date = df['date'].min().strftime(format='%d %B %Y') # start date
    end_date = df['date'].max().strftime(format='%d %B %Y') # end date

    # del nyt, date, articles, materials

    # extract the dates
    dates = list(df['date'].apply(lambda x: x.replace(hour=0, minute=0, second=0, microsecond=0)))

    # extract the text
    headline = list(df['headline'].astype(str))
    snippet = list(df['snippet'].apply(lambda x: x.replace('|', ' ')))
    paragraph = list(df['paragraph'].astype(str))
    texts = [x + ' ' + y + ' ' + z for x, y, z in zip(headline, snippet, paragraph)]

    print('---- creating corpus -----')
    # organize the text
    corpus = list(nlp.pipe(texts))

    X = [] # documents
    Y = [] # terms

    for doc in corpus:

        terms = str()

        for token in doc:

            if token.is_alpha and not token.is_stop:

                terms = terms + ' ' + token.lemma_.lower()

                Y.append(token.lemma_.lower())

        X.append(terms)

    X = np.array(X)
    Y = np.array(Y)

    del corpus, doc, terms, token, texts

    print('---- fitting NMF ----')

    # fit the model
    if selected_model == 'nmf':

        vectorizer = TfidfVectorizer(min_df=10, max_df=0.9)

        x = vectorizer.fit_transform(X)
        y = vectorizer.get_feature_names()

        model = NMF(n_components=n_components, beta_loss=2, alpha=0.1, l1_ratio=0.5, random_state=100)

        fit = model.fit(x)

    elif selected_model == 'plsa':

        vectorizer = TfidfVectorizer(min_df=10, max_df=0.9)

        x = vectorizer.fit_transform(X)
        y = vectorizer.get_feature_names()

        model = NMF(n_components=n_components, beta_loss=1, solver='mu', alpha=0.1, l1_ratio=0.5, random_state=100)

        fit = model.fit(x)

    else:

        vectorizer = CountVectorizer()

        x = vectorizer.fit_transform(X)
        y = vectorizer.get_feature_names()

        model = LatentDirichletAllocation(n_components=n_components, random_state=0)

        fit = model.fit(x)

    # extract the topics and the corresponding keywords
    n_top_words = 200
    n_keywords = 25

    keywords = []

    stop_words = ['xx', 'xxxx']

    for u, v in enumerate(fit.components_):

        # identify the top words based on their relevance
        r = np.zeros(len(v))

        for i in range(len(v)):

            a = v[i] / np.sum(v) # frequency of term "i" in topic "v"
            b = len(Y[Y == y[i]]) / len(Y) # frequency of term "i" in the corpus
            r[i] = np.log(a) - 0.6 * np.log(b) # relevance of term "i" to topic "v"

        top_words = [y[i] for i in r.argsort()[:-n_top_words - 1:-1] if len(y[i]) > 4]

        # extract the keywords from the top words
        tokens = []

        i = 0

        while len(tokens) < n_keywords:

            i += 1

            doc = nlp(top_words[i].capitalize())

            for token in doc:

                if token.lemma_.lower() not in tokens and token.lemma_.lower() not in keywords and \
                token.pos_ != 'ADJ' and token.pos_ != 'VERB' and token.pos_ != 'ADV' \
                and token.ent_type_ == '' and token.lemma_.lower() not in stop_words:

                    tokens.append(token.lemma_.lower())

        keywords.append(tokens)

    # extract the topics
    topics = [x[0] for x in keywords]

    del u, v, top_words, stop_words, token, tokens, doc, r, a, b

    # map the articles to topics
    mapping = fit.transform(x)
    classes = [topics[i] for i in list(np.argmax(mapping, axis=1))]

    # count the number of articles by topic
    counts = pd.Series(classes).value_counts()
    counts = [counts.values[counts.index == x][0] for x in topics]

    del X, Y, x, y, model, fit, vectorizer

    # generate the color palette
    if n_components == 2:

        markercolors = ['rgba' + str(x)[:-1] + ', 0.8)' for x in sequential.Tokyo_3.colors[:-1]]
        fillcolors = ['rgba' + str(x)[:-1] + ', 0.5)' for x in sequential.Tokyo_3.colors[:-1]]

    elif n_components == 3:

        markercolors = ['rgba' + str(x)[:-1] + ', 0.8)' for x in sequential.Tokyo_4.colors[:-1]]
        fillcolors = ['rgba' + str(x)[:-1] + ', 0.5)' for x in sequential.Tokyo_4.colors[:-1]]

    elif n_components == 4:

        markercolors = ['rgba' + str(x)[:-1] + ', 0.8)' for x in sequential.Tokyo_5.colors[:-1]]
        fillcolors = ['rgba' + str(x)[:-1] + ', 0.5)' for x in sequential.Tokyo_5.colors[:-1]]

    elif n_components == 5:

        markercolors = ['rgba' + str(x)[:-1] + ', 0.8)' for x in sequential.Tokyo_6.colors[:-1]]
        fillcolors = ['rgba' + str(x)[:-1] + ', 0.5)' for x in sequential.Tokyo_6.colors[:-1]]

    elif n_components == 6:

        markercolors = ['rgba' + str(x)[:-1] + ', 0.8)' for x in sequential.Tokyo_7.colors[:-1]]
        fillcolors = ['rgba' + str(x)[:-1] + ', 0.5)' for x in sequential.Tokyo_7.colors[:-1]]

    elif n_components == 7:

        markercolors = ['rgba' + str(x)[:-1] + ', 0.8)' for x in sequential.Tokyo_8.colors[:-1]]
        fillcolors = ['rgba' + str(x)[:-1] + ', 0.5)' for x in sequential.Tokyo_8.colors[:-1]]

    elif n_components == 8:

        markercolors = ['rgba' + str(x)[:-1] + ', 0.8)' for x in sequential.Tokyo_9.colors[:-1]]
        fillcolors = ['rgba' + str(x)[:-1] + ', 0.5)' for x in sequential.Tokyo_9.colors[:-1]]

    elif n_components == 9:

        markercolors = ['rgba' + str(x)[:-1] + ', 0.8)' for x in sequential.Tokyo_10.colors[:-1]]
        fillcolors = ['rgba' + str(x)[:-1] + ', 0.5)' for x in sequential.Tokyo_10.colors[:-1]]

    else:

        markercolors = ['rgba' + str(x)[:-1] + ', 0.8)' for x in sequential.Tokyo_11.colors[:-1]]
        fillcolors = ['rgba' + str(x)[:-1] + ', 0.5)' for x in sequential.Tokyo_11.colors[:-1]]

    markercolors = dict(zip(topics, markercolors))
    fillcolors = dict(zip(topics, fillcolors))

    # generate the sunburst chart
    labels = [' ']
    ids = [' ']
    parents = ['']
    colors = ['#fafafa']
    values = [np.sum(counts)]
    widths = [np.max([np.int(n_keywords * counts[i] / np.sum(counts)), 2]) for i in range(len(counts))]

    for i in range(len(topics)):

        labels.extend([topics[i].center(20, ' ')])
        labels.extend([x.center(20, ' ') for x in keywords[i][1:(widths[i] + 1)]])

        ids.extend([topics[i].center(20, ' ')])
        ids.extend([topics[i] + '_' + str(x) for x in range(widths[i])])

        parents.extend([' '])
        parents.extend([topics[i].center(20, ' ')] * widths[i])

        colors.extend([markercolors[topics[i]]])
        colors.extend([markercolors[topics[i]]] * widths[i])

        values.extend([counts[i]])
        values.extend([np.int(counts[i] / widths[i])] * (widths[i] - 1) + [counts[i] - \
        np.int(counts[i] / widths[i]) * (widths[i] - 1)])

    layout = dict(paper_bgcolor='#fafafa',
                  plot_bgcolor='#fafafa',
                  margin=dict(t=0, b=0, l=0, r=0, pad=0),
                  uniformtext=dict(minsize=8, mode='show'))

    traces = go.Sunburst(ids=ids,
                         labels=labels,
                         parents=parents,
                         values=values,
                         marker=dict(colors=colors),
                         insidetextorientation='radial',
                         branchvalues='total',
                         hoverinfo='none',
                         textfont=dict(family='Open Sans'))

    topic_view = go.Figure(data=traces, layout=layout).to_dict()

    del labels, ids, parents, values, colors, widths, layout, traces

    # generate the line chart
    data = pd.DataFrame({'date': dates, 'class': classes, 'count': 1})
    data = pd.pivot_table(data=data, index=['date'], columns=['class'], values=['count'], aggfunc='sum')
    data.fillna(value=0, inplace=True)
    data.columns = [x[1] for x in list(data.columns)]

    layout = dict(plot_bgcolor='white',
                  paper_bgcolor='#fafafa',
                  font=dict(family='Open Sans', color='#737373'),
                  margin=dict(t=10, b=0, l=0, r=0, pad=0),
                  legend=dict(x=0, y=1.15, orientation='h'),
                  xaxis=dict(zeroline=False, showgrid=False, mirror=True, linecolor='#d9d9d9', color='#737373',
                  tickangle=0, tickformat='%d %b <br> (%a)'),
                  yaxis=dict(zeroline=False, showgrid=False, mirror=True, linecolor='#d9d9d9', color='#737373',
                  title='Number of complaints'))

    traces = []

    for i in range(data.shape[1]):

        traces.append(go.Scatter(x=list(data.index),
                                 y=list(data.iloc[:, i]),
                                 name=data.columns[i],
                                 mode='lines',
                                 line=dict(shape='spline', color=fillcolors[data.columns[i]]),
                                 stackgroup='group',
                                 hovertemplate='<b>Topic: </b>' + data.columns[i] + '<br>'
                                 '<b>Date: </b>%{x|%d %b (%a) %Y}<br><b>Number of articles: </b>%{y}<extra></extra>'))

    evolution_view = go.Figure(data=traces, layout=layout).to_dict()

    del data, layout, traces
    
    # save the results
    headline = json.dumps(headline)
    snippet = json.dumps(snippet)
    paragraph = json.dumps(paragraph)

    topics = json.dumps(topics)
    keywords = json.dumps(keywords)
    mapping = json.dumps(mapping.tolist())

    markercolors = json.dumps(markercolors)
    fillcolors = json.dumps(fillcolors)

    articles_data = {'headline': headline, 'snippet': snippet, 'paragraph': paragraph, 'topics': topics,
    'keywords': keywords, 'mapping': mapping, 'markercolors': markercolors, 'fillcolors': fillcolors}

    return [count, start_date, end_date, topic_view, evolution_view, articles_data]

@app.callback([Output('first_line', 'children'), Output('second_line', 'children'), Output('third_line', 'children'),
    Output('document_view', 'figure')], [Input('articles_data', 'children'), Input('articles_counter', 'children')])
def update_document(data, counter):

    # extract the selected article
    if counter is not None:

        i = int(counter)

    else:

        i = 0

    # extract the results for the selected article
    headline = json.loads(data['headline'])
    snippet = json.loads(data['snippet'])
    paragraph = json.loads(data['paragraph'])

    topics = json.loads(data['topics'])
    keywords = json.loads(data['keywords'])
    mapping = json.loads(data['mapping'])

    markercolors = json.loads(data['markercolors'])
    fillcolors = json.loads(data['fillcolors'])

    headline = nlp(headline[i])
    snippet = nlp(snippet[i])
    paragraph = nlp(paragraph[i])

    keywords = np.array(keywords)
    mapping = np.array(mapping)
    mapping = normalize(mapping, norm='l1', axis=1)

    # generate the radar chart for the selected article
    layout=dict(paper_bgcolor='white',
                font=dict(family='Open Sans'),
                margin=dict(t=70, r=70, l=70, b=70, pad=0),
                polar=dict(angularaxis=dict(visible=True,
                                            linecolor='#d9d9d9',
                                            gridcolor='#d9d9d9'),
                           radialaxis=dict(range=[0, 1],
                                           visible=True,
                                           linecolor='#d9d9d9',
                                           gridcolor='#d9d9d9',
                                           showticklabels=False),
                           bgcolor='white'))

    traces = go.Barpolar(r=list(mapping[i]),
                         theta=list(topics),
                         marker_color=list(fillcolors.values()),
                         marker_line_color=list(markercolors.values()),
                         marker_line_width=2,
                         hoverinfo='none')

    document_view = go.Figure(data=traces, layout=layout).to_dict()

    # highlight the keywords in the selected article
    first_line = []

    for token in headline:

        if token.lemma_.lower() in list(keywords.flatten()):

            index = np.where(keywords == token.lemma_.lower())[0][0]
            color = fillcolors[topics[index]]

            first_line.append(html.Div(children=[token.text], style={'display': 'inline', 'color': 'white',
            'background-color': color}))

            first_line.append(html.Div(children=[' '], style={'display': 'inline', 'color': '#212121',
            'background-color': 'white'}))

        else:

            first_line.append(html.Div(children=[token.text], style={'display': 'inline', 'color': '#212121',
            'background-color': 'white'}))

            first_line.append(html.Div(children=[' '], style={'display': 'inline', 'color': '#212121',
            'background-color': 'white'}))

    second_line = []

    for token in snippet:

        if token.lemma_.lower() in list(keywords.flatten()):

            index = np.where(keywords == token.lemma_.lower())[0][0]
            color = fillcolors[topics[index]]

            second_line.append(html.Div(children=[token.text], style={'display': 'inline', 'color': 'white',
            'background-color': color}))

            second_line.append(html.Div(children=[' '], style={'display': 'inline', 'color': '#212121',
            'background-color': 'white'}))

        else:

            second_line.append(html.Div(children=[token.text], style={'display': 'inline', 'color': '#212121',
            'background-color': 'white'}))

            second_line.append(html.Div(children=[' '], style={'display': 'inline', 'color': '#212121',
            'background-color': 'white'}))

    third_line = []

    for token in paragraph:

        if token.lemma_.lower() in list(keywords.flatten()):

            index = np.where(keywords == token.lemma_.lower())[0][0]
            color = fillcolors[topics[index]]

            third_line.append(html.Div(children=[token.text], style={'display': 'inline', 'color': 'white',
            'background-color': color}))

            third_line.append(html.Div(children=[' '], style={'display': 'inline', 'color': '#212121',
            'background-color': 'white'}))

        else:

            third_line.append(html.Div(children=[token.text], style={'display': 'inline', 'color': '#212121',
            'background-color': 'white'}))

            third_line.append(html.Div(children=[' '], style={'display': 'inline', 'color': '#212121',
            'background-color': 'white'}))

    # remove the unnecessary white space
    for i in range(0, len(first_line) - 1):

        if first_line[i + 1].children in [[','], ['.'], [';'], [':'], ['’s'], ['?'], ['\'']]:

            first_line[i].children = ['']

    for i in range(0, len(second_line) - 1):

        if second_line[i + 1].children in [[','], ['.'], [';'], [':'], ['’s'], ['?'], ['\'']]:

            second_line[i].children = ['']

    for i in range(0, len(third_line) - 1):

        if third_line[i + 1].children in [[','], ['.'], [';'], [':'], ['’s'], ['?'], ['\'']]:

            third_line[i].children = ['']

    return [first_line, second_line, third_line, document_view]

@app.callback(Output('articles_counter', 'children'), [Input('previous', 'n_clicks_timestamp'),
    Input('next', 'n_clicks_timestamp')], [State('articles_counter', 'children')])
def update_counter(previous, next, counter):

    if counter is None:

        counter = 0

    if previous is None and next is not None:

        return np.max([counter + 1, 0])

    elif previous is not None and next is None:

        return np.max([counter - 1, 0])

    elif previous is not None and next is not None:

        previous, next = int(previous), int(next)

        if next > previous:

            counter = np.max([counter + 1, 0])

        elif previous > next:

            counter = np.max([counter - 1, 0])

        return counter
    
if __name__ == '__main__':
    app.run_server(debug=False)
