<h2 style="color: purple; text-align:center">Data Mining Group Project </h2>

<h4 style="color: green; text-align:center"><b>Team:</b></h4>
<div >
<ol  style="color: green; text-align:center; list-style:none">
<li>
Coupet, Alexa
</li>
<li>
Marsh, Steven
</li>
<li>
Farley, Douglas</li>
<li>
                    Marzouq, Murtadha 
</li>
</ol>                    
</div>

<h2>Note:</h2> <b> Run The Analysis Python File</b>

 <table border="1" width = "100%">                    <tr>             <td>                <table border = "1" width = "100%">                   <tr>                      <th>Task</th>                      <th>Framework</th>                   </tr>                   <tr>                      <td>Step 1: Loading data</td>                      <td><a href="https://github.com/twintproject/twint">Twint</a></td>                   </tr>                   <tr>                      <td>Step 2: Pre-processing the tweets/articles </td>                      <td><a href="https://spacy.io/usage/v3-2">SpaCy</a></td>   
 <tr>
 <td> Step 3: Data analysis and visualization</td> 
  <td><a href="https://pandas.pydata.org/">Pandas</a></td> 
    <tr>
   <td> Important Step: Topic Modeling</td> 
  <td><a href="https://scikit-learn.org/stable/model_selection.html#model-selection/">pyldavis</a></td> 
         </tr>      
 </tr>
 </tr>                </table>             </td>          </tr>                 </table>

<div class="show-content user_content clearfix enhanced">
  <h1 class="page-title">Tentative Description</h1>
  
  
    <p><strong>Step 1: Loading data (3 pts)</strong></p>
<p>Successfully load the data you choose to analyze into memory.</p>
<p><strong>Step 2: Pre-processing the tweets/articles (8pts total)</strong></p>
<p>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; You can use NLTK or SpaCy (highly recommended) for this task.</p>
<ol>
<li>Tokenization (2pt)</li>
<li>Lemmatization (2pt)</li>
<li>Stopwords removal
<ol>
<li>removing standard stopwords (2pts)</li>
<li>removing additional stopwords as you see fit (2pts)</li>
</ol>
</li>
</ol>
<p><strong>Step 3: Data analysis and visualization (15pts)</strong></p>
<p>Here is a required task to summary your data:</p>
<ul>
<li>“topic modeling” You can follow this post: <a href="https://towardsdatascience.com/topic-model-visualization-using-pyldavis-fecd7c18fbf6" target="_blank" class="external" rel="noreferrer noopener"><span>https://towardsdatascience.com/topic-model-visualization-using-pyldavis-fecd7c18fbf6</span><span aria-hidden="true" class="ui-icon ui-icon-extlink ui-icon-inline" title="Links to an external site."></span><span class="screenreader-only">&nbsp;(Links to an external site.)</span></a>&nbsp;</li>
</ul>
<p>Pick 1 task from the following tasks:&nbsp;</p>
<ul>
<li>“search and count”,</li>
<li>“sentiment analysis”,</li>
<li>"emotion analysis",</li>
<li>“named entity recognition”,</li>
<li>“hashtag analysis”,&nbsp;</li>
</ul>
<p>Note that tasks including “sentiment analysis” and “named entity recognition” should be performed<span>&nbsp;</span><u>before&nbsp;</u>preprocessing for best result.</p>
<p>To be more specific, if you wish to do search and count on the Twitter dataset, you can choose a keyword or a group of keywords (you can use “AND”, “OR”, “NOT” here) that you are interested in.</p>
<p>&nbsp;</p>
<p>If you wish to perform sentiment analysis, get the polarity result for each tweet and then you can choose how to aggregate the sentiment analysis results. One example is that you can aggregate on a daily basis. One recommended method to such aggregation is adding up the positive and negative polarities separately and then divide them by the total number of tweets on that day.</p>
<p><strong>Required output:</strong></p>
<ol>
<li>(5pts) Topic modeling results <a href="https://towardsdatascience.com/topic-model-visualization-using-pyldavis-fecd7c18fbf6" target="_blank" class="external" rel="noreferrer noopener"><span>https://towardsdatascience.com/topic-model-visualization-using-pyldavis-fecd7c18fbf6</span><span aria-hidden="true" class="ui-icon ui-icon-extlink ui-icon-inline" title="Links to an external site."></span><span class="screenreader-only">&nbsp;(Links to an external site.)</span></a> &nbsp;</li>
<li>(5pts) Example tweets/articles for the chosen optional task you performed. For example, if your task is search and count with keyword “immigration”, print out 20 tweets/articles that match your search. If your task is sentiment analysis, print out 20 tweets that have the strongest positive sentiment and 20 tweets that have the strongest negative sentiment. If you task is named entity recognition, print out 10 news articles with its entity results. Printing out such results not only help us understand the dataset better, but also will help validating your python script.</li>
<li><span>(5pts) Pick a question you are interested in asking and provide a simple visualization to answer it. For example, if your question is about trends over time, you can construct a timeline to visualize the trends (i.e. trends on keyword mentions, sentiment change, or entity mentions).</span></li>
</ol>
  
</div>