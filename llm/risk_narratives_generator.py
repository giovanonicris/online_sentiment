import os
import pandas as pd
import openai
from datetime import datetime, timedelta
import requests
from io import StringIO

class EmergingRiskNarrativeGenerator:
    def __init__(self):
        self.api_key = os.getenv('OPENAI_API_KEY_CG')
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY_CG environment variable not set")
        
        self.client = openai.OpenAI(api_key=self.api_key)
        self.csv_url = "https://raw.githubusercontent.com/giovanonicris/online_sentiment/main/output/emerging_risks_online_sentiment.csv"
        
    def load_data_from_github(self):
        # load csv data directly from github
        print("loading data from github...")
        try:
            response = requests.get(self.csv_url)
            response.raise_for_status()
            
            # read csv from the response
            csv_data = StringIO(response.text)
            df = pd.read_csv(csv_data)
            
            print(f"loaded {len(df)} rows from github")
            return df
            
        except Exception as e:
            print(f"error loading data: {e}")
            return None
    
    def filter_last_4_weeks(self, df):
        # filter data to last 4 weeks
        print("filtering to last 4 weeks...")
        
        # assuming there's a date column - adjust column name as needed
        date_columns = [col for col in df.columns if 'date' in col.lower() or 'time' in col.lower()]
        
        if not date_columns:
            print("no date column found. using all data.")
            return df
        
        date_col = date_columns[0]  # use first date column found
        print(f"using date column: {date_col}")
        
        try:
            # convert to datetime
            df[date_col] = pd.to_datetime(df[date_col])
            
            # filter last 4 weeks
            four_weeks_ago = datetime.now() - timedelta(weeks=4)
            filtered_df = df[df[date_col] >= four_weeks_ago]
            
            print(f"filtered to {len(filtered_df)} rows from last 4 weeks")
            return filtered_df
            
        except Exception as e:
            print(f"error filtering dates: {e}. using all data.")
            return df
    
    def group_by_risk_id(self, df):
        # group data by emerging_risk_id
        print("grouping by emerging_risk_id...")
        
        if 'EMERGING_RISK_ID' not in df.columns:
            print("emerging_risk_id column not found")
            print(f"available columns: {list(df.columns)}")
            return None
        
        grouped = df.groupby('EMERGING_RISK_ID')
        risk_groups = {}
        
        for risk_id, group in grouped:
            risk_groups[risk_id] = group
            
        print(f"found {len(risk_groups)} unique emerging risks")
        return risk_groups
    
    def create_narrative_prompt(self, risk_data, risk_id):
        """Create prompt for OpenAI to generate narrative"""
        
        # Get key information from the data
        num_articles = len(risk_data)
        
        # Get sample headlines/summaries if available
        text_columns = [col for col in risk_data.columns if any(word in col.lower() for word in ['title', 'summary', 'content', 'text', 'headline'])]
        
        sample_content = ""
        if text_columns:
            text_col = text_columns[0]
            sample_texts = risk_data[text_col].dropna().head(5).tolist()
            sample_content = "\n".join([f"- {text[:200]}..." for text in sample_texts])
        
        prompt = f"""
        You are an expert risk analyst. Analyze the following emerging risk data from the last 4 weeks and create a comprehensive narrative summary.
        
        EMERGING RISK ID: {risk_id}
        NUMBER OF ARTICLES/MENTIONS: {num_articles}
        TIME PERIOD: Last 4 weeks
        
        SAMPLE CONTENT:
        {sample_content}
        
        Please provide:
        1. **Executive Summary** (2-3 sentences about what this emerging risk represents)
        2. **Recent Developments** (What has happened in the last 4 weeks?)
        3. **Key Themes** (What are the main topics/concerns?)
        4. **Risk Level Assessment** (Low/Medium/High and why)
        5. **Potential Impact** (What could happen if this risk materializes?)
        6. **Recommended Actions** (What should be monitored or done?)
        
        Keep the narrative concise but comprehensive (300-500 words). Focus on actionable insights.
        """
        
        return prompt
    
    def generate_narrative(self, risk_data, risk_id):
        """Generate narrative using OpenAI"""
        prompt = self.create_narrative_prompt(risk_data, risk_id)
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are an expert emerging risk analyst who creates clear, actionable risk narratives for executives."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=800,
                temperature=0.3
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            return f"Error generating narrative for Risk ID {risk_id}: {e}"
    
    def process_all_risks(self, risk_groups):
        """Process all risks and generate narratives"""
        print("Generating AI narratives...")
        
        results = []
        total_risks = len(risk_groups)
        
        for i, (risk_id, risk_data) in enumerate(risk_groups.items(), 1):
            print(f"Processing risk {i}/{total_risks}: ID {risk_id}")
            
            narrative = self.generate_narrative(risk_data, risk_id)
            
            results.append({
                'EMERGING_RISK_ID': risk_id,
                'ARTICLE_COUNT': len(risk_data),
                'AI_NARRATIVE': narrative,
                'GENERATED_DATE': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            })
        
        return results
    
    def save_results(self, results):
        """Save results to CSV in output folder"""
        # Create output folder if it doesn't exist
        output_dir = "../output"  # Save in main output folder
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"emerging_risk_narratives_{timestamp}.csv"
        filepath = os.path.join(output_dir, filename)
        
        # Also save a latest version (overwrites each time)
        latest_filepath = os.path.join(output_dir, "emerging_risk_narratives_latest.csv")
        
        df_results = pd.DataFrame(results)
        df_results.to_csv(filepath, index=False)
        df_results.to_csv(latest_filepath, index=False)
        
        print(f"Results saved to: {filepath}")
        print(f"Latest version: {latest_filepath}")
        return filepath
    
    def run_analysis(self):
        """Main method to run the complete analysis"""
        print("🚀 Starting Emerging Risk Narrative Generation...")
        
        # Load data
        df = self.load_data_from_github()
        if df is None:
            return None
        
        # Filter to last 4 weeks
        df_filtered = self.filter_last_4_weeks(df)
        
        # Group by risk ID
        risk_groups = self.group_by_risk_id(df_filtered)
        if risk_groups is None:
            return None
        
        # Generate narratives
        results = self.process_all_risks(risk_groups)
        
        # Save results
        output_file = self.save_results(results)
        
        print(f"Analysis complete! Generated narratives for {len(results)} emerging risks")
        return output_file

def main():
    try:
        generator = EmergingRiskNarrativeGenerator()
        output_file = generator.run_analysis()
        
        if output_file:
            print(f"\nSuccess! Output file: {output_file}")
        else:
            print("Analysis failed")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
