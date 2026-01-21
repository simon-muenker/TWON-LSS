import statistics
import pandas as pd
import pydantic
import json
import typing
import plotly.graph_objects as go
import plotly.express as px
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
from twon_lss.utility.llm import LLM, EmbeddingModelInterface
import numpy as np
from sklearn.decomposition import PCA
import umap
from typing import List, Optional, Callable, Union, Tuple
from plotly.subplots import make_subplots
from bertopic import BERTopic
import networkx as nx
import matplotlib.pyplot as plt

class RunEvaluation(pydantic.BaseModel):

    path: str
    embedding_model: EmbeddingModelInterface = None
    name: str = pydantic.Field(default_factory=lambda: "Run")
    df: pd.DataFrame = None
    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)
    strip_round_0: bool = True
    feed: list = None

    def model_post_init(self, __context: typing.Any):
        
        data = json.loads(open(self.path+"/feed.json").read())
        self.feed = json.loads(open(self.path+"/feed.json").read())

        results_parsed = []
        for elem in data:
            results_parsed.append({
                "user": elem["user"]["id"],
                "tweet": elem["content"],
                "created_at": elem["timestamp"],
                "like_count": len(elem["likes"]),
                "read_count": len(elem["reads"]),
                "embedding": elem["embedding"] if "embedding" in elem else None
            })

        self.df = pd.DataFrame(results_parsed)

        if (self._has_embeddings() == False) and self.embedding_model:
            self._ensure_embeddings()
        else:
            self._load_embeddings()

        if self.strip_round_0:
            self.df = self.df[self.df["created_at"] > 0]
            self.feed = [post for post in self.feed if post["timestamp"] > 0]

    # Embedding Handling
    def _ensure_embeddings(self):
        if not self._load_embeddings():
            # check for partially missing embeddings
            if self.df["embedding"].isna().any() and self.df["embedding"].notna().any():
                print(f"{self.df['embedding'].isna().sum()} missing embeddings detected. {self.df['embedding'].notna().sum()} embeddings present.")
                self._repair_embeddings()
            else:
                self._generate_embeddings()
            
    def _load_embeddings(self) -> bool:
        try:
            self.df["embedding"] = list(np.load(f"{self.path}/embeddings.npz")["embeddings"])
            print("Found existing embeddings.")
            return True
        except FileNotFoundError:
            pass
        return False

    def _has_embeddings(self) -> bool:
        return not self.df["embedding"].isnull().any()
    
    def _repair_embeddings(self):
        print("Repairing missing embeddings...")

        delete_indices = []

        for i, row in self.df.iterrows():
            # check for well formed embedding
            if row["embedding"] is None or len(row["embedding"]) < 768:
                try:
                    embedding = self.embedding_model.extract([row["tweet"]])[0]
                    self.df.at[i, "embedding"] = embedding
                except Exception as e:
                    print(f"Error generating embedding for tweet {row['tweet']}: {e}")
                    delete_indices.append(i)

        if delete_indices:
            self.df.drop(delete_indices, inplace=True)
            self.df.reset_index(drop=True, inplace=True)

        # drop rows from feed.json that still have no embedding
        if input(f"Delete {len(delete_indices)} posts from {self.path} that could not be embedded? (y/n): ") == "y":
            data = json.loads(open(self.path).read())
            data_repaired = [elem for i, elem in enumerate(data) if i not in delete_indices]
            with open(self.path, "w") as f:
                json.dump(data_repaired, f, indent=4)
        else:
            print("Skipping deletion of posts from feed.json. Current DataFrame will have fewer posts than feed.json and not be reloadable.")
            return

        # Save embeddings
        np.savez(f"{self.path}/embeddings.npz",
                embeddings=np.array(self.df["embedding"].to_list())
        )
        print(f"Saved repaired embeddings to {self.path}/embeddings.npz")

    def _generate_embeddings(self):

        print("Generating embeddings...")
        
        embeddings: list = self.embedding_model.extract(self.df["tweet"].tolist()) # There is apparently a limit on the number of texts that can be processed at once
        self.df["embedding"] = embeddings

        # Save embeddings
        np.savez(f"{self.path}/embeddings.npz",
                embeddings=np.array(self.df["embedding"].to_list())
        )
        print(f"Saved embeddings to {self.path}/embeddings.npz")

    # Plotting Utilities
    def _make_figure(
        self,
        traces: List[go.Scatter] | go.Scatter,
        title: str,
        xaxis_title: str,
        yaxis_title: str
    ) -> go.Figure:
        """Wrap traces in a figure with layout."""
        fig = go.Figure()
        
        if isinstance(traces, go.Scatter):
            traces = [traces]
        
        for trace in traces:
            fig.add_trace(trace)
        
        fig.update_layout(
            title=title,
            xaxis_title=xaxis_title,
            yaxis_title=yaxis_title
        )
        return fig
    
    def classify_and_plot_distribution(
    self,
    classifier: dict,
    return_fig: bool = True
    ) -> Union[go.Figure, Tuple[pd.DataFrame, List[go.Bar]]]:
        """Classify posts and return distribution data/plot."""

        label_dist = self._get_label_distribution(classifier)
        traces = self._create_bar_traces(label_dist, name="Label Distribution")

        if return_fig:
            return self._make_figure(traces, "Label Distribution", "Label", "Proportion")
        return label_dist, traces

    def _get_label_distribution(self, classifier: dict, normalize: bool = True) -> pd.DataFrame:
        """Get normalized label distribution. Handles both single-label and multi-label."""
        predictions = classifier["classifier"].predict(self.df["embedding"].tolist())
        
        if classifier.get("multi_label", False):
            # Multi-label: count documents containing each label
            label_counts = {}
            for pred in predictions:
                for i, label_idx in enumerate(pred):
                    if label_idx == 1:
                        label_name = classifier["label_names"][i]
                        label_counts[label_name] = label_counts.get(label_name, 0) + 1

            n_docs = len(self.df)
            label_dist = pd.DataFrame([
                {"label": label, "proportion": count / n_docs if normalize else count}
                for label, count in sorted(label_counts.items(), key=lambda x: x[1], reverse=True)
            ])
        else:
            # Single-label: standard value counts
            self.df["label"] = [classifier["label_names"][pred] for pred in predictions]
            label_dist = self.df["label"].value_counts(normalize=normalize).reset_index()
            label_dist.columns = ["label", "proportion"]
        
        return label_dist

    def _create_bar_traces(
        self, 
        label_dist: pd.DataFrame, 
        name: str, 
        color: str = None
    ) -> List[go.Bar]:
        """Create bar trace from label distribution."""
        trace_kwargs = {
            "x": label_dist["label"], 
            "y": label_dist["proportion"], 
            "name": name
        }
        if color:
            trace_kwargs["marker_color"] = color
        return [go.Bar(**trace_kwargs)]


    # Analyzing individual Runs
    def generate_graph(self):
        """Generate interaction graph from feed data."""
        G = nx.Graph()
        for post in self.feed:
            user = post["user"]["id"]
            G.add_node(user)
            for user_read in post["reads"]:
                if user_read["id"] not in G:
                    G.add_node(user_read["id"])
                if not G.has_edge(user, user_read["id"]):
                    G.add_edge(user, user_read["id"], weight=1)
                else:
                    G[user][user_read["id"]]["weight"] += 1
        return G
    
    def visualize_graph(self, G=None):
        G = G if G else self.generate_graph()
        fig, ax = plt.subplots(figsize=(10, 10))
        pos = nx.spring_layout(G, seed=42)
        
        # Draw edges - thin and grey
        nx.draw_networkx_edges(G, pos, width=0.5, edge_color='grey', alpha=0.05, ax=ax)
        
        # Draw nodes - blue
        nx.draw_networkx_nodes(G, pos, node_size=20, ax=ax)
        
        ax.axis('off')
        plt.tight_layout()
        plt.show()

    def describe(self, log: bool = True, print_stats: bool = True):
        
        # Basic statistics
        n_posts = len(self.df)
        n_users = self.df["user"].nunique()
        n_views = sum(self.df["read_count"])

        # Average, Median and STD of views and posts per user and per post
        posts_per_user = self.df.groupby("user").size()
        views_per_user = self.df.groupby("user")["read_count"].sum()
        views_per_post = self.df["read_count"]

        # Network statistics based on read interactions
        G = self.generate_graph()

        stats = {
            "n_posts": n_posts,
            "n_users": n_users,
            "n_views": n_views,
            "avg_posts_per_user": posts_per_user.mean(),
            "median_posts_per_user": posts_per_user.median(),
            "std_posts_per_user": posts_per_user.std(),
            "avg_views_per_user": views_per_user.mean(),
            "median_views_per_user": views_per_user.median(),
            "std_views_per_user": views_per_user.std(),
            "avg_views_per_post": views_per_post.mean(),
            "median_views_per_post": views_per_post.median(),
            "std_views_per_post": views_per_post.std(),
            "avg_degree": np.mean([d for n, d in G.degree()]),
            "median_degree": statistics.median([d for n, d in G.degree()]),
            "std_degree": statistics.stdev([d for n, d in G.degree()]),
            "n_connected_components": nx.number_connected_components(G),
            "avg_clustering_coefficient": nx.average_clustering(G)
        }

        if log:
            path = self.path.replace("json", "_description.json")
            with open(path, "w") as f:
                json.dump(stats, f, indent=4)
            print(f"Saved description to {path}")

        if print_stats:
            print(f"Description of Run: {self.name}")
            for key, value in stats.items():
                print(f"  {key}: {value}")        

        return stats


    def consistency_per_user(self) -> pd.DataFrame:
        """Compute consistency of embeddings per user."""
        user_consistency = {}

        for user in self.df["user"].unique():
            user_embeddings = np.array(self.df[self.df["user"] == user]["embedding"].tolist())
            if len(user_embeddings) > 1:
                sim_matrix = cosine_similarity(user_embeddings)
                # Take upper triangle without diagonal
                upper_tri_indices = np.triu_indices_from(sim_matrix, k=1)
                sims = sim_matrix[upper_tri_indices]
                user_consistency[user] = np.mean(sims)
            else:
                user_consistency[user] = None  # Not enough data to compute consistency

        consistency_df = pd.DataFrame.from_dict(user_consistency, orient='index', columns=['consistency'])
        return consistency_df
    
    def plot_consistency_distribution(self) -> go.Figure:
        """Plot distribution of user consistency."""
        consistency_df = self.consistency_per_user()
        consistency_df = consistency_df.dropna()

        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=consistency_df['consistency'],
            nbinsx=30,
            name='User Consistency',
            marker_color='blue',
            opacity=0.75
        ))

        fig.update_layout(
            title='Distribution of User Consistency',
            xaxis_title='Consistency (Average Cosine Similarity)',
            yaxis_title='Count',
            bargap=0.2
        )

        return fig
    
    
    @classmethod
    def compare_consistency_between_runs(
        cls,
        runs: List["RunEvaluation"]
    ) -> go.Figure:
        """Compare consistency distributions between multiple runs."""
        fig = go.Figure()

        for run in runs:
            consistency_df = run.consistency_per_user()
            consistency_df = consistency_df.dropna()

            fig.add_trace(go.Histogram(
                x=consistency_df['consistency'],
                nbinsx=30,
                name=run.name,
                opacity=0.5
            ))

        fig.update_layout(
            title='Comparison of User Consistency Between Runs',
            xaxis_title='Consistency (Average Cosine Similarity)',
            yaxis_title='Count',
            barmode='overlay',
            bargap=0.2
        )

        return fig   


    # Comparing Runs
    @classmethod
    def describe_runs(
        cls,
        runs: List["RunEvaluation"],
    ) -> pd.DataFrame:
        """Compare basic statistics of multiple runs."""
        stats_df = pd.DataFrame([
            {**run.describe(log=False, print_stats=False), "run_name": run.name}
            for run in runs
        ])

        # Reorder columns to put run_name first
        cols = ["run_name"] + [col for col in stats_df.columns if col != "run_name"]
        stats_df = stats_df[cols]
            
        return stats_df
    
    @classmethod
    def compare_runs_content(
        cls,
        runs: List["RunEvaluation"],
        classifiers: List[dict],
    ) -> go.Figure:
        """Compare runs with grouped bars per classifier."""
        
        # Assign consistent colors to each run
        colors = px.colors.qualitative.Plotly
        run_colors = {run.name: colors[i % len(colors)] for i, run in enumerate(runs)}

        fig = make_subplots(
            rows=len(classifiers), 
            cols=1,
            subplot_titles=[f"{classifiers[i]['classifier_name']}" for i in range(len(classifiers))]
        )

        for j, classifier in enumerate(classifiers):
            for run in runs:
                label_dist = run._get_label_distribution(classifier)
                traces = run._create_bar_traces(
                    label_dist, 
                    name=run.name, 
                    color=run_colors[run.name]
                )

                for trace in traces:
                    trace.legendgroup = run.name
                    trace.showlegend = (j == 0)
                    fig.add_trace(trace, row=j+1, col=1)

        fig.update_layout(barmode='group')
        fig.update_yaxes(title_text="Proportion")

        fig.update_layout(
            height=400 * len(classifiers),
            title_text="Run Comparison Across Classifiers",
            showlegend=True,
            legend_title="Runs"
        )

        return fig

    @classmethod
    def compare_runs_content_over_time(
        cls,
        runs: List["RunEvaluation"],
        classifiers: Callable[[str], str],
        running_average: bool = True,
        window_size: int = 3
    ) -> go.Figure:
        """Compare runs over time for a given classifier."""

        # Define color palette for runs and dash patterns for labels
        colors = px.colors.qualitative.Plotly  # or any other color sequence
        run_colors = {run.name: colors[i % len(colors)] for i, run in enumerate(runs)}

        dash_styles = ['solid', 'dash', 'dot', 'dashdot', 'longdash', 'longdashdot']

        # Collect all unique labels across all runs/classifiers to create consistent mapping
        all_labels = set()
        for run in runs:
            for classifier in classifiers:
                df = run.df.copy()
                labels = classifier.predict(df["embedding"].tolist())
                all_labels.update(labels)

        label_dash = {label: dash_styles[i % len(dash_styles)] for i, label in enumerate(sorted(all_labels))}

        fig = make_subplots(
            rows=len(classifiers), 
            cols=1,
            subplot_titles=[f"Classifier {i+1}" for i in range(len(classifiers))]
        )

        for j, classifier in enumerate(classifiers):
            for run in runs:
                df = run.df.copy()
                df["label"] = list(classifier.predict(df["embedding"].tolist()))

                df_time = df.groupby(["created_at", "label"]).size().reset_index(name='share')
                df_time['share'] = df_time.groupby('created_at')['share'].transform(lambda x: x / x.sum())

                if running_average:
                    df_time['share'] = df_time.groupby('label')['share'].transform(
                        lambda x: x.rolling(window=window_size, min_periods=1).mean()
                    )

                labels = df_time["label"].unique()

                for label in labels:
                    df_label = df_time[df_time["label"] == label]
                    trace = go.Scatter(
                        x=df_label["created_at"],
                        y=df_label["share"],
                        mode='lines+markers',
                        name=f"{run.name} - {label}",
                        line=dict(color=run_colors[run.name], dash=label_dash[label]),
                        marker=dict(color=run_colors[run.name])
                    )
                    trace.legendgroup = run.name
                    trace.showlegend = (j == 0)
                    fig.add_trace(trace, row=j+1, col=1)

        fig.update_yaxes(title_text="Share")  # Changed from "Count" since you're plotting shares

        return fig

    @classmethod
    def compare_runs_topics(
                    cls,    
                   runs: List["RunEvaluation"],
                   n_topics: int = 10
                   ):
        
        docs = []
        embeddings = []
        for run in runs:
            docs.extend(run.df["tweet"].tolist())
            embeddings.extend(run.df["embedding"].tolist())

        topic_model = BERTopic()
        topics, probs = topic_model.fit_transform(docs, embeddings=np.array(embeddings))
        
        fig = go.Figure()

        offset = 0
        for run in runs:
            run_len = len(run.df)
            run.df["label"] = topics[offset:offset + run_len]
            offset += run_len
            
            # Filter to top n topics
            df = run.df[(run.df["label"] != -1) & (run.df["label"] < n_topics)]

            # Replace labels with topic names
            df["label"] = df["label"].apply(lambda x: topic_model.get_topic_info(x).iloc[0]["Name"])

            traces = run._create_bar_traces(
                df["label"].value_counts(normalize=True).reset_index(),
                name=run.name
            )
            for trace in traces:
                fig.add_trace(trace)

        # Layout
        fig.update_layout(
            title="Topic Distribution Comparison",
            xaxis_title="Topic",
            yaxis_title="Proportion",
            barmode='group'
        )
        return fig





    ### Deprecated Methods for Individual Runs
    def plot_views_over_time(self, return_fig: bool = True, cumulative: bool = False) -> Union[go.Figure, List[go.Scatter]]:
        traces = []
        
        for username in self.df["user"].unique():
            df_user = self.df[self.df["user"] == username].copy()
            df_user = df_user.groupby("created_at").sum("read_count").reset_index()
            df_user = df_user.sort_values("created_at")

            if cumulative:
                df_user["read_count"] = df_user["read_count"].cumsum()

            traces.append(go.Scatter(
                x=df_user["created_at"],
                y=df_user["read_count"],
                name=username
            ))
        
        if return_fig:
            return self._make_figure(traces, "View Count per User Over Time", "Time Step", "View Count")
        return traces

    # Per Dataset Plots
    def classify_and_plot_distribution_over_time(
        self,
        classifier: Callable[[str], str],
        return_fig: bool = True
    ) -> Union[go.Figure, List[go.Scatter]]:
        """
        Classify posts using the provided classifier and plot the distribution over time.
        """
        
        traces = []
        
        self.df["label"] = list(classifier.predict(self.df["embedding"].tolist()))
        
        df_time = self.df.groupby(["created_at", "label"]).size().reset_index(name='counts')
        labels = df_time["label"].unique()
        
        for label in labels:
            df_label = df_time[df_time["label"] == label]
            traces.append(go.Scatter(
                x=df_label["created_at"],
                y=df_label["counts"],
                mode='lines+markers',
                name=str(label)
            ))
        
        if return_fig:
            return self._make_figure(traces, "Label Distribution Over Time", "Time Step", "Count")
        return traces