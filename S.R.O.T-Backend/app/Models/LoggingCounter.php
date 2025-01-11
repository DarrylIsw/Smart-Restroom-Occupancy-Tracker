<?php

namespace App\Models;

use Illuminate\Database\Eloquent\Factories\HasFactory;
use Illuminate\Database\Eloquent\Model;

class LoggingCounter extends Model
{
    use HasFactory;
    protected $fillable = ['toilet_id', 'floor_id', 'building_id', 'gender_id', 'usage_count', 'logged_at', 'is_reset'];
    
    // Relationship with Toilet
    public function toilet()
    {
      return $this->belongsTo(Toilet::class, 'toilet_id');
    }
    
    // Relationship with Floor
    public function floor()
    {
      return $this->belongsTo(Floor::class, 'floor_id');
    }
    
    // Relationship with Building
    public function building()
    {
      return $this->belongsTo(Building::class, 'building_id');
    }

    // Relationship with Gender
    public function gender()
    {
      return $this->belongsTo(Building::class, 'gender_id');
    }
}
