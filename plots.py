import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from wordcloud import WordCloud
import random


NAMES = ['Simón', 'Marcos', 'Luz', 'Mónica y Pablo']

def stacked_barchart(df: pd.DataFrame) -> go.Figure:
    # Create chart_df
    chart_df = (
        df.groupby(['Name', 'IsSeries'])
        ['Title']
        .count()
        .reset_index()
    )

    # Order chart_df
    chart_df['order'] = chart_df['Name'].apply(lambda n: NAMES.index(n))
    chart_df = chart_df.sort_values(by='order')

    # Populate data to plot
    series = chart_df.loc[chart_df['IsSeries']]
    movies = chart_df.loc[~chart_df['IsSeries']]

    data = []
    data.append(go.Bar(
        name='Series',
        x=series['Name'],
        y=series['Title'],
        marker_color = '#024a70',
        text=series['Title'],
        showlegend=True
    ))

    data.append(go.Bar(
        name='Movies',
        x=movies['Name'],
        y=movies['Title'],
        marker_color = '#74d0f0',
        text=movies['Title'],
        showlegend=True
    ))
    
    # Prepare layout
    layout = go.Layout(
        barmode='stack',
        xaxis_title=None,
        yaxis_title='# Titles Watched',
        title_text='Titles Watched in Netflix, by Family Member',
        height=450,
        width=950
    )

    # Build figure
    fig = go.Figure(
        data=data,
        layout=layout
    )

    return fig


def donut_chart(df: pd.DataFrame) -> go.Figure:
    # Create chart_df
    chart_df = (
        df.groupby(['Name', 'IsSeries'])
        ['Title']
        .count()
        .reset_index()
    )

    # Order chart_df
    chart_df['order'] = chart_df['Name'].apply(lambda n: NAMES.index(n))
    chart_df = chart_df.sort_values(by='order')

    # Add type
    chart_df['Type'] = chart_df['IsSeries'].apply(lambda x: 'Series' if x else 'Movies')

    fig = make_subplots(
        rows=1, cols=len(NAMES) + 1,
        # column_widths=[0.6, 0.4],
        # row_heights=[0.4, 0.6],
        specs=[[{'type': 'domain'}] * (len(NAMES) + 1)]
    )

    for name in NAMES:
        name_chart_df = chart_df.loc[chart_df['Name'] == name]
        
        fig.add_trace(
            go.Pie(
                labels=name_chart_df['Type'],
                values=name_chart_df['Title'], 
                marker={
                    'colors': ['#74d0f0', '#024a70']
                },
                title=f'{name}',
                # titlefont='bold',
                # sort=False,
                titleposition='top center',
                textinfo='percent',
                textsrc='bold',
                hole=.50,
            ),
            row=1, col=NAMES.index(name) + 1
        )
        
    fig.update_layout(
        title_text='Series & Movies Distribution, by Family Member',
        showlegend=False,
        height=450,
        width=950
    )

    return fig


class BubbleChart:
    def __init__(self, area, bubble_spacing=0):
        """
        Setup for bubble collapse.

        Parameters
        ----------
        area : array-like
            Area of the bubbles.
        bubble_spacing : float, default: 0
            Minimal spacing between bubbles after collapsing.

        Notes
        -----
        If "area" is sorted, the results might look weird.
        """
        area = np.asarray(area)
        r = np.sqrt(area / np.pi)

        self.bubble_spacing = bubble_spacing
        self.bubbles = np.ones((len(area), 4))
        self.bubbles[:, 2] = r
        self.bubbles[:, 3] = area
        self.maxstep = 2 * self.bubbles[:, 2].max() + self.bubble_spacing
        self.step_dist = self.maxstep / 2

        # calculate initial grid layout for bubbles
        length = np.ceil(np.sqrt(len(self.bubbles)))
        grid = np.arange(length) * self.maxstep
        gx, gy = np.meshgrid(grid, grid)
        self.bubbles[:, 0] = gx.flatten()[:len(self.bubbles)]
        self.bubbles[:, 1] = gy.flatten()[:len(self.bubbles)]

        self.com = self.center_of_mass()

    def center_of_mass(self):
        return np.average(
            self.bubbles[:, :2], axis=0, weights=self.bubbles[:, 3]
        )

    def center_distance(self, bubble, bubbles):
        return np.hypot(bubble[0] - bubbles[:, 0],
                        bubble[1] - bubbles[:, 1])

    def outline_distance(self, bubble, bubbles):
        center_distance = self.center_distance(bubble, bubbles)
        return center_distance - bubble[2] - \
            bubbles[:, 2] - self.bubble_spacing

    def check_collisions(self, bubble, bubbles):
        distance = self.outline_distance(bubble, bubbles)
        return len(distance[distance < 0])

    def collides_with(self, bubble, bubbles):
        distance = self.outline_distance(bubble, bubbles)
        return np.argmin(distance, keepdims=True)

    def collapse(self, n_iterations=50):
        """
        Move bubbles to the center of mass.

        Parameters
        ----------
        n_iterations : int, default: 50
            Number of moves to perform.
        """
        for _i in range(n_iterations):
            moves = 0
            for i in range(len(self.bubbles)):
                rest_bub = np.delete(self.bubbles, i, 0)
                # try to move directly towards the center of mass
                # direction vector from bubble to the center of mass
                dir_vec = self.com - self.bubbles[i, :2]

                # shorten direction vector to have length of 1
                dir_vec = dir_vec / np.sqrt(dir_vec.dot(dir_vec))

                # calculate new bubble position
                new_point = self.bubbles[i, :2] + dir_vec * self.step_dist
                new_bubble = np.append(new_point, self.bubbles[i, 2:4])

                # check whether new bubble collides with other bubbles
                if not self.check_collisions(new_bubble, rest_bub):
                    self.bubbles[i, :] = new_bubble
                    self.com = self.center_of_mass()
                    moves += 1
                else:
                    # try to move around a bubble that you collide with
                    # find colliding bubble
                    for colliding in self.collides_with(new_bubble, rest_bub):
                        # calculate direction vector
                        dir_vec = rest_bub[colliding, :2] - self.bubbles[i, :2]
                        dir_vec = dir_vec / np.sqrt(dir_vec.dot(dir_vec))
                        # calculate orthogonal vector
                        orth = np.array([dir_vec[1], -dir_vec[0]])
                        # test which direction to go
                        new_point1 = (self.bubbles[i, :2] + orth *
                                      self.step_dist)
                        new_point2 = (self.bubbles[i, :2] - orth *
                                      self.step_dist)
                        dist1 = self.center_distance(
                            self.com, np.array([new_point1]))
                        dist2 = self.center_distance(
                            self.com, np.array([new_point2]))
                        new_point = new_point1 if dist1 < dist2 else new_point2
                        new_bubble = np.append(new_point, self.bubbles[i, 2:4])
                        if not self.check_collisions(new_bubble, rest_bub):
                            self.bubbles[i, :] = new_bubble
                            self.com = self.center_of_mass()

            if moves / len(self.bubbles) < 0.1:
                self.step_dist = self.step_dist / 2

    def plot(self, ax, labels, colors):
        """
        Draw the bubble plot.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
        labels : list
            Labels of the bubbles.
        colors : list
            Colors of the bubbles.
        """
        for i in range(len(self.bubbles)):
            circ = plt.Circle(
                self.bubbles[i, :2], self.bubbles[i, 2], color=colors[i])
            ax.add_patch(circ)
            ax.text(*self.bubbles[i, :2], labels[i],
                    horizontalalignment='center', verticalalignment='center')


def buble_chart(df: pd.DataFrame):
    # Build chart_df
    chart_df = (
        df
        .loc[(df['Name'] == 'Simón') & (df['IsSeries'])]
        .groupby(['Title']) # , 'IsSeries', 'Name'])
        [['Title']]
        .count()
        .rename(columns={'Title': 'Count'})
        .reset_index()
        .sort_values(by='Count', ascending=False)
    )

    # chart_df['Type'] = chart_df['IsSeries'].apply(lambda x: 'Series' if x else 'Movies')

    # chart_df['order'] = chart_df['Name'].apply(lambda n: NAMES.index(n))
    # chart_df = chart_df.sort_values(by=['order', 'Count'], ascending=[True, False]).drop(columns=['order'])

    chart_df.loc[
        (chart_df['Count'] <= 50)
        # & (chart_df['IsSeries'])
        , ['Title']
    ] = 'Others'

    others = chart_df.loc[chart_df['Title'] == 'Others']

    chart_df = chart_df.loc[chart_df['Title'] != 'Others']

    # chart_df = pd.concat([
    #     chart_df, 
    #     pd.DataFrame({
    #         'Title': 'Others',
    #         'Count': others['Count'].sum()
    #     }, index=[0])
    # ], axis=0)

    max_size = 15
    chart_df['Count'] = (chart_df['Count'] / chart_df['Count'].max()) * max_size

    chart_df['Color'] = 'lightgrey' # chart_df['Title'].apply(lambda x: "#{:06x}".format(random.randint(0, 0xFFFFFF)))

            
    bubble_chart = BubbleChart(
        area=chart_df['Count'],
        bubble_spacing=0.5
    )

    bubble_chart.collapse()

    fig, ax = plt.subplots(figsize=(12,8), subplot_kw=dict(aspect="equal"))

    bubble_chart.plot(
        ax, 
        chart_df['Title'].to_list(), 
        chart_df['Color'].to_list()
    )

    ax.axis("off")
    ax.relim()
    ax.autoscale_view()
    ax.set_title('Favourite Series - Simón')

    return fig


def stacked_area_chart(df: pd.DataFrame) -> go.Figure:
    # Create subplots: one for each name
    fig = make_subplots(
        rows=len(NAMES), cols=1,  # Multiple rows, 1 column
        specs=[[{'type': 'xy'}]] * len(NAMES),  # Use 'xy' for scatter plots
        subplot_titles=[f'{name}: Monthly Movies & Series' for name in NAMES]
    )

    # Loop through the names and create stacked area charts for each
    for name in NAMES:
        chart_df = df.loc[df['Name'] == name]
        
        # Series data
        series_per_month = (
            chart_df
            .loc[chart_df['IsSeries']]
            .groupby(pd.Grouper(key='Date', freq='M'))
            [['Title']]
            .count()
            .rename(columns={'Title': 'Series'})
        )
        
        # Movies data
        movies_per_month = (
            chart_df
            .loc[~chart_df['IsSeries']]
            .groupby(pd.Grouper(key='Date', freq='M'))
            [['Title']]
            .count()
            .rename(columns={'Title': 'Movies'})
        )
        
        # Combine both series and movies, filling missing months with 0
        combined_data = pd.concat([series_per_month, movies_per_month], axis=1).fillna(0)
        combined_data['Movies'] = combined_data['Series'] + combined_data['Movies']
        
        # Add series trace
        fig.add_trace(
            go.Scatter(
                # title=f'{name} Series & Movies',
                x=combined_data.index, 
                y=combined_data['Series'],
                mode='lines',
                # name=f'{name} Series',
                fill='tozeroy',  # Fill to the x-axis (stacking starts here)
                line=dict(color='#024a70'),
                showlegend=False
            ),
            row=NAMES.index(name) + 1, col=1
        )
        
        # Add movies trace, stacked on top of series
        fig.add_trace(
            go.Scatter(
                x=combined_data.index, 
                y=combined_data['Movies'],
                mode='lines',
                # name=f'{name} Movies',
                fill='tonexty',  # Fill to the previous trace
                line=dict(color='#74d0f0'),
                showlegend=False
            ),
            row=NAMES.index(name) + 1, col=1
        )
        
        # Fit a linear regression model with the quadratic terms
        combined_data['DateNumeric'] = pd.to_numeric(combined_data.index.to_series().apply(lambda x: x.toordinal()))
        X = combined_data[['DateNumeric']]
        y = combined_data['Movies']
        
        model = LinearRegression()
        model.fit(X, y)
        
        combined_data['Trend'] = model.predict(X)
        
        # Add trend trace
        fig.add_trace(
            go.Scatter(
                x=combined_data.index, 
                y=combined_data['Trend'],
                mode='lines',
                line=dict(color='grey', width=2, dash='dash'),
                showlegend=False
            ),
            row=NAMES.index(name) + 1, col=1
        )

    # Update layout
    fig.update_layout(
        title_text='Number of Titles over Time, by Family Member',
        height=1700,
        width=1300
    )

    return fig


def waterfall_chart(df: pd.DataFrame) -> go.Figure:
    # Create subplots: one for each name
    fig = make_subplots(
        rows=len(NAMES), cols=1,  # Multiple rows, 1 column
        specs=[[{'type': 'xy'}]] * len(NAMES),  # Use 'xy' for scatter plots
        subplot_titles=[f'{name}: Avg Titles per Day' for name in NAMES]
    )

    # Loop through the names and create stacked area charts for each
    for name in NAMES:
        chart_df = df.loc[df['Name'] == name]
        chart_df['DayOfWeek'] = chart_df['Date'].dt.day_name()

        div = (df['Date'].max() - df['Date'].min()).days / 7

        chart_df = (
            chart_df
            .groupby(['Name', 'DayOfWeek'])
            [['Title']]
            .count()
            .reset_index()
        )

        DOW = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        chart_df['order'] = chart_df['DayOfWeek'].apply(lambda d: DOW.index(d))
        chart_df = chart_df.sort_values(by='order')
        chart_df.drop(columns=['order'], inplace=True)

        chart_df['Title'] = chart_df['Title'] / div
        chart_df.rename(columns={'Title': 'Avg Titles per Day'}, inplace=True)

        chart_df = pd.concat([
            chart_df, 
            pd.DataFrame({
                'Name': name,
                'DayOfWeek': 'All',
                'Avg Titles per Day': chart_df['Avg Titles per Day'].sum()
            }, index=[0])
        ], axis=0)

        chart_df['Measure'] = ['relative'] * 7 + ['total']
        
        # Add series trace
        fig.add_trace(
            go.Waterfall(
                name=name, 
                orientation="v",
                measure=chart_df['Measure'],
                x=chart_df['DayOfWeek'],
                textposition="outside",
                text=chart_df['Avg Titles per Day'].apply(lambda x: round(x, 2)),
                y=chart_df['Avg Titles per Day'],
                connector={"line":{"color":"rgb(63, 63, 63)"}},
            ),
            row=NAMES.index(name) + 1, col=1
        )

    # Update layout
    fig.update_layout(
        title = "Avg Titles Seen Per Day",
        showlegend=False,
        height=1700,
        width=1000
    )

    return fig


def treemap(df: pd.DataFrame, others_n: int = 10) -> go.Figure:
    # Create chart_df
    chart_df = (
        df
        .groupby(['Title', 'IsSeries', 'Name'])
        [['Title']]
        .count()
        .rename(columns={'Title': 'Count'})
        .reset_index()
        .sort_values(by='Count', ascending=False)
    )

    chart_df['Type'] = chart_df['IsSeries'].apply(lambda x: 'Series' if x else 'Movies')

    chart_df['Titles'] = 'Titles'

    chart_df['order'] = chart_df['Name'].apply(lambda n: NAMES.index(n))
    chart_df = chart_df.sort_values(by=['order', 'Count'], ascending=[True, False]).drop(columns=['order'])

    chart_df.loc[
        (chart_df['Count'] <= others_n) &
        (chart_df['IsSeries'])
        , ['Title']
    ] = 'Others'

    # Build treemap
    fig = px.treemap(
        chart_df, 
        path=['Titles', 'Name', 'Type', 'Title'], 
        values='Count'
    )

    fig.update_traces(root_color="lightgrey")

    fig.update_layout(
        title='Series & Movies Treemap',
        height=1500,
        width=1200,
        margin = dict(t=50, l=25, r=25, b=25)
    )

    return fig


def series_wordcloud(df: pd.DataFrame) -> go.Figure:
    # Extract Text
    series = df.loc[(df['IsSeries'])]
    text = " ".join(title for title in series.Title)

    # Generate the word cloud
    width=1000
    height=800
    wordcloud = WordCloud(width=width, height=height, background_color='white').generate(text)

    # Save the word cloud image
    # wordcloud.to_file("wordcloud.png")

    # Read the image using Plotly
    fig = px.imshow(wordcloud, title="Series Word Cloud")
    fig.update_layout(
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        width=width, height=height
    )

    return fig


def movies_wordcloud(df: pd.DataFrame) -> go.Figure:
    # Extract Text
    movies = df.loc[(~df['IsSeries'])]
    text = " ".join(title for title in movies.Title)

    # Generate the word cloud
    width=1000
    height=800
    wordcloud = WordCloud(width=width, height=height, background_color='white').generate(text)

    # Save the word cloud image
    # wordcloud.to_file("wordcloud.png")
    
    # Read the image using Plotly
    fig = px.imshow(wordcloud, title="Movies Word Cloud")
    fig.update_layout(
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        width=width, height=height
        
    )

    return fig