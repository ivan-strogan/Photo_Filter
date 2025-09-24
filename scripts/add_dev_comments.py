#!/usr/bin/env python3
"""
Script to help add developer-friendly comments to code.

This script demonstrates good commenting practices for junior developers.
"""

def example_well_commented_function(data: list, threshold: float = 0.5) -> dict:
    """
    Example of a well-commented function for junior developers.

    This function shows how to write clear, helpful comments that explain
    not just WHAT the code does, but WHY and HOW it works.

    Args:
        data: List of numerical values to process
        threshold: Minimum value to include (default: 0.5)

    Returns:
        Dictionary with processed results

    For junior developers:
    - Notice how we explain the algorithm step by step
    - Comments explain WHY we make certain choices
    - Variable names are descriptive
    - Edge cases are handled and documented
    """
    # Validate input data - always check your inputs!
    if not data:
        # Early return for empty data - this is called "guard clause" pattern
        return {"error": "No data provided", "count": 0}

    # Filter data above threshold - list comprehension is more pythonic
    # than traditional for loops for simple filtering
    filtered_data = [x for x in data if x >= threshold]

    # Calculate statistics - we compute these in separate steps for clarity
    # Even though we could do it in one line, readable code is better
    total_count = len(filtered_data)

    if total_count > 0:
        # Only calculate average if we have data - avoid division by zero
        average = sum(filtered_data) / total_count
        max_value = max(filtered_data)
        min_value = min(filtered_data)
    else:
        # Set sensible defaults when no data meets criteria
        average = max_value = min_value = 0.0

    # Return structured data - using a dictionary makes results easy to use
    return {
        "count": total_count,
        "average": average,
        "max": max_value,
        "min": min_value,
        "threshold_used": threshold,
        "original_count": len(data)
    }

def show_commenting_best_practices():
    """
    Examples of good vs bad commenting practices.

    This function demonstrates what makes comments helpful vs unhelpful.
    """

    # ❌ BAD COMMENT - states the obvious
    x = 5  # Set x to 5

    # ✅ GOOD COMMENT - explains why
    x = 5  # Use 5-second timeout to avoid hanging on slow networks

    # ❌ BAD COMMENT - out of date
    y = x * 2  # Calculate area (this used to calculate area, now it's doubled timeout)

    # ✅ GOOD COMMENT - explains the business logic
    timeout_seconds = x * 2  # Double the base timeout for retry operations

    # ❌ BAD COMMENT - too verbose for simple code
    # This loop iterates through each number in the list
    # and checks if the number is greater than 10
    # and if it is, adds it to the results list
    results = []
    for num in [1, 5, 12, 8, 15]:
        if num > 10:
            results.append(num)

    # ✅ GOOD COMMENT - explains the purpose, not the mechanics
    # Filter for numbers that exceed our minimum threshold for processing
    results = [num for num in [1, 5, 12, 8, 15] if num > 10]

    return results

class ExampleClassWithComments:
    """
    Example class showing good commenting practices.

    This class demonstrates how to document classes, methods, and complex logic
    in a way that helps junior developers understand the code.

    For junior developers:
    - Class docstring explains the purpose and how to use it
    - Each method is documented with purpose, args, and returns
    - Complex algorithms have step-by-step comments
    - We explain design decisions and trade-offs
    """

    def __init__(self, name: str, max_items: int = 100):
        """
        Initialize the example class.

        Args:
            name: Human-readable name for this instance
            max_items: Maximum items to store (prevents memory issues)

        For junior developers:
        - __init__ is Python's constructor method
        - We validate inputs and set sensible defaults
        - Instance variables (self.x) belong to this specific object
        """
        # Validate required parameters
        if not name or not isinstance(name, str):
            raise ValueError("Name must be a non-empty string")

        # Store configuration
        self.name = name
        self.max_items = max_items

        # Initialize data storage - using list for ordered data
        self.items = []

        # Track statistics - these will be updated as we add items
        self.total_added = 0
        self.total_rejected = 0

    def add_item(self, item: any) -> bool:
        """
        Add an item to our collection with size limits.

        This method demonstrates error handling, logging, and
        maintaining data integrity.

        Args:
            item: Any item to add to our collection

        Returns:
            True if item was added, False if rejected

        For junior developers:
        - Methods that modify state should return success/failure
        - We check preconditions before modifying data
        - Statistics are updated consistently
        """
        # Check if we have space - this prevents memory issues
        if len(self.items) >= self.max_items:
            # Track rejections for debugging/monitoring
            self.total_rejected += 1
            return False

        # Add the item - list.append() is O(1) operation
        self.items.append(item)

        # Update statistics - keep these in sync with data changes
        self.total_added += 1

        return True

    def get_stats(self) -> dict:
        """
        Get current statistics about our collection.

        Returns:
            Dictionary with current stats

        For junior developers:
        - This is a "getter" method that doesn't modify state
        - We compute some values dynamically (current_size)
        - Return format is consistent and easy to use
        """
        return {
            "name": self.name,
            "current_size": len(self.items),          # Computed dynamically
            "max_capacity": self.max_items,           # From configuration
            "total_added": self.total_added,          # Running counter
            "total_rejected": self.total_rejected,    # Running counter
            "utilization": len(self.items) / self.max_items if self.max_items > 0 else 0
        }

if __name__ == "__main__":
    print("This script demonstrates good commenting practices for junior developers.")
    print("Look at the source code to see examples of helpful comments!")

    # Example usage
    example = ExampleClassWithComments("Demo Collection", max_items=5)

    # Add some test data
    for i in range(7):  # Try to add more than max_items to test limits
        success = example.add_item(f"item_{i}")
        print(f"Adding item_{i}: {'✅' if success else '❌'}")

    # Show final statistics
    stats = example.get_stats()
    print(f"\nFinal stats: {stats}")