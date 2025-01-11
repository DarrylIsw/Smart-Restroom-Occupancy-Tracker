<?php

namespace App\Http\Controllers;

use App\Models\Floor;
use App\Models\Building;
use App\Models\Toilet;
use Illuminate\Support\Facades\DB;
use App\Models\LoggingCounter;
use Illuminate\Http\Request;

class FloorController extends Controller
{
    /**
     * Display a listing of the resource.
     */
    public function index($buildingId)
    {
        $floors = Floor::where('building_id', $buildingId)->with('toilets.gender')->get();
        return response()->json($floors);
    }

    public function addFloor(Request $request, $buildingId)
    {
        // Validate input
        $request->validate([
            'floor_number' => 'required|integer|min:1',
        ]);

        // Check if the building exists
        $building = Building::findOrFail($buildingId);

        // Check for existing floor with the same building_id and floor_number
        $existingFloor = Floor::where('building_id', $building->id)
                              ->where('floor_number', $request->floor_number)
                              ->first();

        if ($existingFloor) {
            return response()->json(['error' => 'Floor number already exists for this building.'], 400);
        }

        // Create the new floor
        $floor = new Floor();
        $floor->building_id = $building->id;
        $floor->floor_number = $request->floor_number;
        $floor->save();

        // Add male and female toilets for the new floor
        $genders = ['Male', 'Female'];
        foreach ($genders as $genderLabel) {
            $genderId = DB::table('genders')->where('label', $genderLabel)->value('id');
            
            if ($genderId) {
                Toilet::create([
                    'floor_id' => $floor->id,
                    'gender_id' => $genderId,
                    'usage_count' => 0,
                ]);
            }
        }

        return response()->json(['success' => 'Floor and toilets added successfully.', 'floor' => $floor->load('toilets')], 201);
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
        // Find the floor
        $floor = Floor::findOrFail($id);
        
        // Delete all associated toilets and logs
        foreach ($floor->toilets as $toilet) {
            LoggingCounter::where('toilet_id', $toilet->id)->delete(); // Delete logs
            $toilet->delete(); // Delete toilets
        }
        
        // Delete the floor itself
        $floor->delete();
        
        return response()->json(['message' => 'Floor and related data deleted successfully.']);
    }
}
