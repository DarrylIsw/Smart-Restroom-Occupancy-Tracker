<?php

namespace App\Http\Controllers;

use App\Models\LoggingCounter;
use App\Models\Toilet;
use Illuminate\Http\Request;

class LoggingCounterController extends Controller
{
    /**
     * Display a listing of the resource.
     */
    public function index()
    {
        return LoggingCounter::with(['toilet', 'floor', 'building', 'gender'])->get();
    }

    public function logSpecificToilet(Request $request, $toiletId)
    {
        $toilet = Toilet::with('floor.building')->findOrFail($toiletId);
    
        LoggingCounter::create([
            'toilet_id' => $toilet->id,
            'floor_id' => $toilet->floor_id,
            'building_id' => optional($toilet->floor->building)->id,
            'usage_count' => $toilet->usage_count,
            'gender_id' => $toilet->gender_id,
            'logged_at' => now(),
        ]);
    
        return response()->json(['message' => 'Toilet usage count logged successfully.']);
    }

    public function getLoggingCounters()
    {
        $logs = LoggingCounter::with([
            'toilet.gender',
            'toilet.floor.building',
        ])->get();
    
        $data = $logs->map(function ($log) {
            return [
                'id' => $log->id,
                'toilet_id' => $log->toilet_id,
                'usage_count' => $log->usage_count,
                'logged_at' => $log->logged_at,
                'building_name' => $log->toilet->floor->building->name,
                'floor_number' => $log->toilet->floor->floor_number,
                'gender' => $log->toilet->gender->label,
            ];
        });
    
        return response()->json($data);
    }

    public function resetUsageCount(Request $request, $toiletId)
    {
        try {
            // Fetch the toilet by ID
            $toilet = Toilet::with('floor.building')->findOrFail($toiletId);
    
            // Log the current usage count before resetting
            LoggingCounter::create([
                'toilet_id' => $toilet->id,
                'floor_id' => $toilet->floor_id,
                'building_id' => optional($toilet->floor->building)->id,
                'usage_count' => $toilet->usage_count,
                'logged_at' => now(),
                'is_reset' => true, // Mark this action as a reset
            ]);
    
            // Reset the usage count
            $toilet->update(['usage_count' => 0]);
    
            return response()->json(['message' => 'Usage count reset and logged successfully.']);
        } catch (\Exception $e) {
            return response()->json(['error' => 'Failed to reset usage count: ' . $e->getMessage()], 500);
        }
    }

    public function clearAllLogs()
    {
        try {
            LoggingCounter::truncate(); // Deletes all entries in the table
            return response()->json(['message' => 'All logs cleared successfully.']);
        } catch (\Exception $e) {
            return response()->json(['error' => 'Failed to clear logs: ' . $e->getMessage()], 500);
        }
    }

    public function deleteSpecificLog(Request $request, $id)
    {
        try {
            // Find the log entry by its ID
            $log = LoggingCounter::findOrFail($id);
            
            // Delete the log entry
            $log->delete();
            
            return response()->json(['message' => 'Log entry deleted successfully.']);
        } catch (\Exception $e) {
            return response()->json(['error' => 'Failed to delete log entry: ' . $e->getMessage()], 500);
        }
    }

    /**
     * Store a newly created resource in storage.
     */
    public function store(Request $request)
    {
        //
    }

    /**
     * Display the specified resource.
     */
    public function show(string $id)
    {
        //
    }

    /**
     * Update the specified resource in storage.
     */
    public function update(Request $request, string $id)
    {
        //
    }

    /**
     * Remove the specified resource from storage.
     */
    public function destroy(string $id)
    {
        //
    }
}
