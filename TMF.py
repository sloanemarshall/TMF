
import pandas as pd
import os
from typing import List, Dict, Any, Union
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TradeMonitor:
    def __init__(self, trades_config: Dict[str, List[Dict[str, Any]]], percentile_rank_list: List[str]):
        """
        Initialize the TradeMonitor with trade configurations and percentile rank lists.
        
        Args:
            trades_config: Dictionary containing trade configurations
            percentile_rank_list: List of assets for percentile rank calculation
        """
        self.trades_config = trades_config
        self.percentile_rank_list = percentile_rank_list
        self.final_df = pd.DataFrame()
        
    def rolling_correlation(self, df: pd.DataFrame, date_col: str, desc_col: str, value_col: str, 
                          series1: str, series2: str, window_size: int, col_desc: str) -> pd.DataFrame:
        """
        Calculate rolling correlation between two series.
        
        Args:
            df: Input DataFrame
            date_col: Date column name
            desc_col: Description column name
            value_col: Value column name
            series1: First series name
            series2: Second series name
            window_size: Window size for rolling correlation
            col_desc: Output column description
            
        Returns:
            DataFrame with rolling correlation results
        """
        try:
            corr_df = df[(df[desc_col] == series1) | (df[desc_col] == series2)]
            corr_df = corr_df.pivot(index=date_col, columns=desc_col, values=value_col).reset_index()
            corr_df[date_col] = pd.to_datetime(corr_df[date_col])
            corr_df = corr_df.sort_values(by=date_col).reset_index(drop=True)
            
            # Check if both series exist in the pivoted dataframe
            if series1 not in corr_df.columns or series2 not in corr_df.columns:
                logger.warning(f"Missing series in correlation calculation: {series1} or {series2}")
                return pd.DataFrame()  # Return empty DataFrame
                
            corr_df[col_desc] = corr_df[series1].rolling(window=window_size).corr(corr_df[series2])
            corr_df = corr_df[[date_col, col_desc]]
            melted_diff = pd.melt(corr_df, id_vars=[date_col], var_name=desc_col, value_name=value_col)
            melted_diff[date_col] = pd.to_datetime(melted_diff[date_col])
            melted_diff[date_col] = melted_diff[date_col].dt.strftime('%m/%d/%Y')
            
            return melted_diff
        except Exception as e:
            logger.error(f"Error in rolling_correlation: {e}")
            return pd.DataFrame()
    
    def diff_calc(self, df: pd.DataFrame, date_col: str, desc_col: str, value_col: str, 
                series1: str, series2: str, col_desc: str) -> pd.DataFrame:
        """
        Calculate difference between two series.
        
        Args:
            df: Input DataFrame
            date_col: Date column name
            desc_col: Description column name
            value_col: Value column name
            series1: First series name
            series2: Second series name
            col_desc: Output column description
            
        Returns:
            DataFrame with difference calculation results
        """
        try:
            diff_df = df[(df[desc_col] == series1) | (df[desc_col] == series2)]
            diff_df = diff_df.pivot(index=date_col, columns=desc_col, values=value_col).reset_index()
            diff_df[date_col] = pd.to_datetime(diff_df[date_col])
            diff_df = diff_df.sort_values(by=date_col).reset_index(drop=True)
            
            # Check if both series exist in the pivoted dataframe
            if series1 not in diff_df.columns or series2 not in diff_df.columns:
                logger.warning(f"Missing series in diff calculation: {series1} or {series2}")
                return pd.DataFrame()
                
            diff_df[col_desc] = diff_df[series1] - diff_df[series2]
            diff_df = diff_df[[date_col, col_desc]]
            melted_diff = pd.melt(diff_df, id_vars=[date_col], var_name=desc_col, value_name=value_col)
            melted_diff[date_col] = pd.to_datetime(melted_diff[date_col])
            melted_diff[date_col] = melted_diff[date_col].dt.strftime('%m/%d/%Y')
            
            return melted_diff
        except Exception as e:
            logger.error(f"Error in diff_calc: {e}")
            return pd.DataFrame()
    
    def math_subtract(self, df: pd.DataFrame, date_col: str, desc_col: str, value_col: str, 
                    series1: str, number: int, col_desc: str) -> pd.DataFrame:
        """
        Subtract series values from a constant number.
        
        Args:
            df: Input DataFrame
            date_col: Date column name
            desc_col: Description column name
            value_col: Value column name
            series1: Series name
            number: Constant number for subtraction
            col_desc: Output column description
            
        Returns:
            DataFrame with subtraction results
        """
        try:
            diff_df = df[df[desc_col] == series1]
            diff_df = diff_df.pivot(index=date_col, columns=desc_col, values=value_col).reset_index()
            diff_df[date_col] = pd.to_datetime(diff_df[date_col])
            diff_df = diff_df.sort_values(by=date_col).reset_index(drop=True)
            
            if series1 not in diff_df.columns:
                logger.warning(f"Missing series in math_subtract: {series1}")
                return pd.DataFrame()
                
            diff_df[col_desc] = number - diff_df[series1]
            diff_df = diff_df[[date_col, col_desc]]
            melted_diff = pd.melt(diff_df, id_vars=[date_col], var_name=desc_col, value_name=value_col)
            melted_diff[date_col] = pd.to_datetime(melted_diff[date_col])
            melted_diff[date_col] = melted_diff[date_col].dt.strftime('%m/%d/%Y')
            
            return melted_diff
        except Exception as e:
            logger.error(f"Error in math_subtract: {e}")
            return pd.DataFrame()
    
    def index_to_100(self, df: pd.DataFrame, date_col: str, desc_col: str, value_col: str, 
                   series: str, new_column_name: str, reindex_date: str = None) -> pd.DataFrame:
        """
        Reindex series to 100 at a specified date.
        
        Args:
            df: Input DataFrame
            date_col: Date column name
            desc_col: Description column name
            value_col: Value column name
            series: Series name
            new_column_name: New column name for indexed values
            reindex_date: Date to reindex to 100
            
        Returns:
            DataFrame with indexed values
        """
        try:
            index_df = df[df[desc_col] == series]
            index_df = index_df.pivot(index=date_col, columns=desc_col, values=value_col).reset_index()
            index_df[date_col] = pd.to_datetime(index_df[date_col]).dt.date
            index_df = index_df.sort_values(by=date_col).reset_index(drop=True)
            
            if series not in index_df.columns:
                logger.warning(f"Missing series in index_to_100: {series}")
                return pd.DataFrame()
            
            # Initialize with 100
            index_df[new_column_name] = 100
            
            # Calculate cumulative returns
            for i in range(1, len(index_df)):
                previous_value = index_df.loc[i-1, new_column_name]
                current_return = index_df.loc[i, series]
                index_df.loc[i, new_column_name] = (1 + current_return) * previous_value
            
            # Reindex to 100 at specified date if provided
            if reindex_date:
                reindex_date = pd.to_datetime(reindex_date).date()
                if reindex_date in index_df[date_col].values:
                    base_value = index_df.loc[index_df[date_col] == reindex_date, new_column_name].values[0]
                    index_df[new_column_name] = (index_df[new_column_name] / base_value) * 100
                else:
                    logger.warning(f"Reindex date '{reindex_date}' not found in the date column.")
                    
            index_df = index_df[[date_col, new_column_name]]
            melted_index = pd.melt(index_df, id_vars=[date_col], var_name=desc_col, value_name=value_col)
            
            melted_index[date_col] = pd.to_datetime(melted_index[date_col])
            melted_index[date_col] = melted_index[date_col].dt.strftime('%m/%d/%Y')
            
            return melted_index
        except Exception as e:
            logger.error(f"Error in index_to_100: {e}")
            return pd.DataFrame()
    
    def calculate_total_return(self, df: pd.DataFrame, date_col: str, desc_col: str, value_col: str, 
                             series: str, start_date: str, new_col_name: str, trade: str) -> pd.DataFrame:
        """
        Calculate total return from a start date.
        
        Args:
            df: Input DataFrame
            date_col: Date column name
            desc_col: Description column name
            value_col: Value column name
            series: Series name
            start_date: Start date for return calculation
            new_col_name: New column name for return value
            trade: Trade name
            
        Returns:
            DataFrame with total return calculation
        """
        try:
            return_df = df[df[desc_col] == series]
            
            return_df = return_df.pivot(index=date_col, columns=desc_col, values=value_col).reset_index()
            return_df[date_col] = pd.to_datetime(return_df[date_col]).dt.date
            return_df = return_df.sort_values(by=date_col).reset_index(drop=True)
            
            if series not in return_df.columns:
                logger.warning(f"Missing series in calculate_total_return: {series}")
                return pd.DataFrame()
                
            start_date = pd.to_datetime(start_date).date()
            
            if start_date not in return_df[date_col].values:
                logger.warning(f"Start Date '{start_date}' not found in the date column")
                return pd.DataFrame()
            
            return_df = return_df[return_df[date_col] >= start_date]
            
            if return_df.empty:
                logger.warning("Empty return dataframe after filtering by start date")
                return pd.DataFrame()
                
            initial_value = return_df.loc[return_df[date_col] == start_date, series].values[0]
            
            # Filter out zero values and get the last non-zero value
            non_zero_df = return_df[return_df[series] != 0]
            if non_zero_df.empty:
                logger.warning("No non-zero values found in series")
                return pd.DataFrame()
                
            end_value = non_zero_df[series].iloc[-1]
            total_return = ((end_value - initial_value) / initial_value) * 100
            
            data = {
                'dates': ['1/1/1900'],  # Using a placeholder date
                'Description': [new_col_name],
                'Values': [total_return],
                'Model': [trade]
            }
            
            return pd.DataFrame(data)
        except Exception as e:
            logger.error(f"Error in calculate_total_return: {e}")
            return pd.DataFrame()
    
    def calculate_percentile_rank(self, df: pd.DataFrame, date_col: str, desc_col: str, value_col: str, 
                                series: str) -> pd.DataFrame:
        """
        Calculate percentile rank of a series.
        
        Args:
            df: Input DataFrame
            date_col: Date column name
            desc_col: Description column name
            value_col: Value column name
            series: Series name
            
        Returns:
            DataFrame with percentile rank calculation
        """
        try:
            df_filtered = df[df[desc_col] == series]
            
            df_filtered = df_filtered.dropna(subset=[date_col, value_col])
            
            if df_filtered.empty:
                logger.warning(f"No data found for series: {series}")
                return pd.DataFrame()
            
            df_pivot = df_filtered.pivot(index=date_col, columns=desc_col, values=value_col).reset_index()
            df_pivot[date_col] = pd.to_datetime(df_pivot[date_col]).dt.date
            df_pivot = df_pivot.sort_values(by=date_col).reset_index(drop=True)
            
            if series not in df_pivot.columns:
                logger.warning(f"Series {series} not found in pivoted data")
                return pd.DataFrame()
                
            rank_col_name = f"{series}_percentile_rank"
            df_pivot[rank_col_name] = df_pivot[series].rank(pct=True)
            
            df_result = df_pivot[[date_col, rank_col_name]]
            
            melted_df = pd.melt(df_result, id_vars=[date_col], var_name=desc_col, value_name=value_col)
            return melted_df
        except Exception as e:
            logger.error(f"Error in calculate_percentile_rank: {e}")
            return pd.DataFrame()
    
    def load_data(self, tmf_input_file: str, systematic_input_file: str, 
                macro_input_file: str = None) -> pd.DataFrame:
        """
        Load data from input files.
        
        Args:
            tmf_input_file: Path to TMF input file
            systematic_input_file: Path to systematic input file
            macro_input_file: Path to macro input file
            
        Returns:
            Combined DataFrame
        """
        try:
            worksheet_list = ['FactSet', 'BloombergDaily', 'Ranks', 'BloombergMonthly', 'BloombergDaily2']
            df_list = []
            
            # Load TMF data
            with pd.ExcelFile(tmf_input_file) as excel_file:
                for sheet in worksheet_list:
                    if sheet in excel_file.sheet_names:
                        df = pd.read_excel(excel_file, sheet)
                        if 'dates' in df.columns:
                            melted_df = pd.melt(df, id_vars='dates', var_name='Description', value_name='Values')
                            df_list.append(melted_df)
                        else:
                            logger.warning(f"Sheet {sheet} does not have 'dates' column")
            
            # Load systematic data
            try:
                systematic_df = pd.read_excel(systematic_input_file)
                model_filter = ['Canada LC vs US LC', 'U.S. Stock Bond']
                filtered_systematic_df = systematic_df[
                    systematic_df['Model'].isin(model_filter) & 
                    (systematic_df['IsSignal'] == 1)
                ]
                filtered_systematic_df = filtered_systematic_df[['dates', 'Values', 'Description', 'Model']]
                df_list.append(filtered_systematic_df)
            except Exception as e:
                logger.error(f"Error loading systematic data: {e}")
            
            # Load macro data if provided
            if macro_input_file:
                try:
                    macro_df = pd.read_excel(macro_input_file, sheet_name='Sheet1')
                    macro_region_filter = ['United States']
                    macro_measure_filter = ['CPI', 'PMI']
                    
                    macro_filtered = macro_df[
                        macro_df['Region'].isin(macro_region_filter) &
                        macro_df['Measure'].isin(macro_measure_filter) &
                        (macro_df['Metric'] == 'Median')
                    ]
                    
                    macro_filtered = macro_filtered[['Dates', 'Value', 'Historical', 'Name', 'Region']]
                    macro_filtered = macro_filtered.rename(columns={
                        'Dates': 'dates', 
                        'Value': 'Values', 
                        'Name': 'Description'
                    })
                    
                    df_list.append(macro_filtered)
                except Exception as e:
                    logger.error(f"Error loading macro data: {e}")
            
            # Combine all data
            final_df = pd.concat(df_list, ignore_index=True)
            
            # Format dates
            final_df['dates'] = pd.to_datetime(final_df['dates'], errors='coerce')
            # Filter out rows with invalid dates
            final_df = final_df.dropna(subset=['dates'])
            final_df['dates'] = final_df['dates'].dt.strftime('%m/%d/%Y')
            
            return final_df
        except Exception as e:
            logger.error(f"Error in load_data: {e}")
            return pd.DataFrame()
    
    def process_data(self) -> pd.DataFrame:
        """
        Process all trade data calculations.
        
        Returns:
            Processed DataFrame
        """
        try:
            # Get sharepoint location with OS-agnostic path handling
            aat_sharepoint_location = os.path.expanduser('~/Edward Jones/Asset Allocation - Documents/')
            
            # Define input files
            tmf_input_file = os.path.join(aat_sharepoint_location, 'Production/OAA Trade Monitoring Framework/Trade_monitoring.xlsx')
            systematic_input_file = os.path.join(aat_sharepoint_location, 'Production/systematic.xlsx')
            macro_input_file = os.path.join(aat_sharepoint_location, 'Production/macro.xlsx')
            
            # Load data
            self.final_df = self.load_data(tmf_input_file, systematic_input_file, macro_input_file)
            
            if self.final_df.empty:
                logger.error("Failed to load data. Empty DataFrame.")
                return pd.DataFrame()
            
            # Process difference calculations
            diff_dfs = []
            for trade in self.trades_config['trades']:
                diff_df = self.diff_calc(
                    self.final_df, 'dates', 'Description', 'Values',
                    trade['Pair 1 Returns'],
                    trade['Pair 2 Returns'],
                    trade['Diff Column Name']
                )
                if not diff_df.empty:
                    diff_dfs.append(diff_df)
            
            if diff_dfs:
                diff_combined = pd.concat(diff_dfs, ignore_index=True)
                self.final_df = pd.concat([self.final_df, diff_combined], ignore_index=True)
            
            # Process index calculations
            index_dfs = []
            for trade in self.trades_config['trades']:
                index_df = self.index_to_100(
                    self.final_df, 'dates', 'Description', 'Values',
                    trade['Diff Column Name'],
                    trade['Relative Return Index Column'],
                    reindex_date=trade['Implementation Date']
                )
                if not index_df.empty:
                    index_dfs.append(index_df)
            
            if index_dfs:
                index_combined = pd.concat(index_dfs, ignore_index=True)
                self.final_df = pd.concat([self.final_df, index_combined], ignore_index=True)
            
            # Process rolling correlation calculations
            corr_dfs = []
            for trade in self.trades_config['trades']:
                corr_df = self.rolling_correlation(
                    self.final_df, 'dates', 'Description', 'Values',
                    trade['Correlation Pair 1'], 
                    trade['Correlation Pair 2'],
                    window_size=trade['Window Size'],
                    col_desc=trade['Correl Description Name']
                )
                if not corr_df.empty:
                    corr_dfs.append(corr_df)
            
            if corr_dfs:
                corr_combined = pd.concat(corr_dfs, ignore_index=True)
                self.final_df = pd.concat([self.final_df, corr_combined], ignore_index=True)
            
            # Process total return calculations
            tr_dfs = []
            for trade in self.trades_config['trades']:
                tr_ipc = self.calculate_total_return(
                    self.final_df, 'dates', 'Description', 'Values',
                    trade['Relative Return Index Column'],
                    trade['IPC Date'], 
                    trade['IPC Label'],
                    trade['trade']
                )
                if not tr_ipc.empty:
                    tr_dfs.append(tr_ipc)
                
                tr_impl = self.calculate_total_return(
                    self.final_df, 'dates', 'Description', 'Values',
                    trade['Relative Return Index Column'],
                    trade['Implementation Date'], 
                    trade['Implementation Label'],
                    trade['trade']
                )
                if not tr_impl.empty:
                    tr_dfs.append(tr_impl)
            
            if tr_dfs:
                tr_combined = pd.concat(tr_dfs, ignore_index=True)
                self.final_df = pd.concat([self.final_df, tr_combined], ignore_index=True)
            
            # Process math subtract
            fed_funds_df = self.math_subtract(
                self.final_df, 'dates', 'Description', 'Values', 
                'FFZ5 Index', 100, 'Fed Funds Futures Pricing'
            )
            if not fed_funds_df.empty:
                self.final_df = pd.concat([self.final_df, fed_funds_df], ignore_index=True)
            
            # Process percentile rank calculations
            rank_dfs = []
            for series in self.percentile_rank_list:
                rank_df = self.calculate_percentile_rank(
                    self.final_df, 'dates', 'Description', 'Values', series
                )
                if not rank_df.empty:
                    rank_dfs.append(rank_df)
            
            if rank_dfs:
                rank_combined = pd.concat(rank_dfs, ignore_index=True)
                self.final_df = pd.concat([self.final_df, rank_combined], ignore_index=True)
            
            return self.final_df
        except Exception as e:
            logger.error(f"Error in process_data: {e}")
            return pd.DataFrame()
    
    def save_data(self, output_file: str = 'output.csv') -> None:
        """
        Save processed data to file.
        
        Args:
            output_file: Output file path
        """
        try:
            if not self.final_df.empty:
                self.final_df.to_csv(output_file, index=False)
                logger.info(f"Data successfully saved to {output_file}")
            else:
                logger.error("No data to save")
        except Exception as e:
            logger.error(f"Error saving data: {e}")


def main():
    # Define trade configurations
    trades_dict = {
        'trades': [
            {
                'trade': 'US LC vs CAN LC',
                'IPC Label': 'Since IPC (10/24/2023)',
                'IPC Date': '10/24/2023',
                'Implementation Label': 'Since Implementation (11/21/2023)',
                'Implementation Date': '11/21/2023',
                'Relative Return Index Column': 'USLC CANLC Rel Index',
                'Pair 1 Returns': 'US LC (CAD) TR',
                'Pair 2 Returns': 'CAN LC TR',
                'Diff Column Name': 'US LC - CAN LC TR Diff',
                'Correlation Pair 1': 'US LC - CAN LC TR Diff',
                'Correlation Pair 2': 'Crude Oil Return', 
                'Window Size': 60,
                'Correl Description Name': 'USLC - CANLC & Oil Rolling Correl'
            }
        ]
    }

    percentile_rank_list = [
        'NVDA Equity',
        'CONSSENT Index',
        '.AAIIB2b G index',
        'MSXAMSIL Index'
    ]

    # Create TradeMonitor instance and process data
    trade_monitor = TradeMonitor(trades_dict, percentile_rank_list)
    processed_df = trade_monitor.process_data()
    
    # Save processed data
    trade_monitor.save_data('output.csv')


if __name__ == "__main__":
    main()